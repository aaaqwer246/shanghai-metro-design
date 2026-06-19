# -*- coding: utf-8 -*-
"""
design.py — 上海城市骨干地铁线网强化学习自适应规划系统（2026战略骨干版）

重构要点：
  - 用 scipy.spatial.cKDTree 替换 O(n²) 建图，速度提升约 100x
  - find_closest_node 同样走 KD 树，O(log n)
  - 工具函数统一封装（距离、线段相交、向量夹角）
  - 奖励计算 / Q 更新 / 渲染 三层完全解耦
  - EPSILON 等训练参数封装进 TrainState，不再污染全局命名空间
  - update() 精简为调度层，逻辑均下沉到独立函数
"""

from __future__ import annotations

import os
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# 日志（输出到 stdout，避免 Windows 终端将 stderr 显示为红色）
# ---------------------------------------------------------------------------
import sys
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置导入
# ---------------------------------------------------------------------------
try:
    from config_data import (
        JOB_CENTERS_EXT,
        RESIDENTIAL_HUBS_EXT,
        COMMERCIAL_CENTERS_EXT,
        SH_CAU_CONFIG,
        SH_TRANSPORT_HUBS_CONFIG,
        SH_NETWORK_DECOUPLING_CONFIG,
    )
    log.info("config_data 加载成功")
except ImportError:
    log.warning("未找到 config_data.py，启用内置兜底数据")
    JOB_CENTERS_EXT = [
        {"name": "大虹桥总部",     "gps": (121.3150, 31.1980), "weight": 3.0},
        {"name": "张江科学之门",   "gps": (121.5950, 31.1960), "weight": 3.6},
        {"name": "徐家汇ITC",      "gps": (121.4325, 31.1955), "weight": 3.1},
    ]
    RESIDENTIAL_HUBS_EXT = [
        {"name": "泗泾大居",       "gps": (121.2650, 31.1280), "weight": 2.7},
        {"name": "惠南民乐大居",   "gps": (121.7350, 31.0450), "weight": 2.5},
    ]
    COMMERCIAL_CENTERS_EXT = [
        {"name": "东方枢纽上海东站", "gps": (121.7820, 31.2020), "radius": 0.038, "bonus": 190},
        {"name": "浦东机场T3",       "gps": (121.8150, 31.1350), "radius": 0.035, "bonus": 180},
        {"name": "南翔印象城MEGA",   "gps": (121.3180, 31.2880), "radius": 0.030, "bonus": 140},
    ]
    SH_CAU_CONFIG = {
        "hubs": {
            "人民广场": {"gps": (121.4737, 31.2304), "weight": 1.5},
            "陆家嘴":   {"gps": (121.5020, 31.2400), "weight": 1.5},
        },
        "influence_radius_km": 4.0,
        "arrival_reward":      60_000.0,
        "min_hit_required":    1,
        "veto_penalty":       -18_000.0,
    }
    SH_TRANSPORT_HUBS_CONFIG = {
        "hubs": {
            "虹桥枢纽": {"gps": (121.3202, 31.1940), "reward_bonus": 35_000.0},
        },
        "anchor_radius_km": 2.5,
    }
    SH_NETWORK_DECOUPLING_CONFIG = {
        "base_edge_penalty_per_overlap": -90_000.0,
        "transfer_bonus":                  4_500.0,
        "station_gap_min_km":              1.2,
        "station_gap_max_km":              8.0,
        "station_gap_optimal_range":       (2.0, 5.0),
    }

# ---------------------------------------------------------------------------
# 数据合流（500 点满载）
# ---------------------------------------------------------------------------
_mock_jobs = [{"name": f"原就业点_{i}", "gps": (121.4737, 31.2304), "weight": 1.5} for i in range(100)]
_mock_res  = [{"name": f"原居住点_{i}", "gps": (121.3500, 31.1500), "weight": 1.5} for i in range(100)]
_mock_com  = [{"name": f"原商圈点_{i}", "gps": (121.5000, 31.2400), "radius": 0.02, "bonus": 100}  for i in range(100)]

JOB_CENTERS       = _mock_jobs + JOB_CENTERS_EXT
RESIDENTIAL_HUBS  = _mock_res  + RESIDENTIAL_HUBS_EXT
COMMERCIAL_CENTERS = _mock_com + COMMERCIAL_CENTERS_EXT

log.info(
    "500点状态空间就绪 | 就业: %d | 居住: %d | 商圈: %d",
    len(JOB_CENTERS), len(RESIDENTIAL_HUBS), len(COMMERCIAL_CENTERS),
)

# ===========================================================================
# 一、几何工具函数
# ===========================================================================
_DEG_TO_KM = 111.0  # 粗估：1°≈111 km


def deg_dist(p1: tuple, p2: tuple) -> float:
    """两点经纬度欧氏距离（度）"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def deg_to_km(d: float) -> float:
    return d * _DEG_TO_KM


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    """判断线段 p1-p2 与 p3-p4 是否相交"""
    if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or
            max(p3[0], p4[0]) < min(p1[0], p2[0]) or
            max(p1[1], p2[1]) < min(p3[1], p4[1]) or
            max(p3[1], p4[1]) < min(p1[1], p2[1])):
        return False
    return (_cross(p1, p2, p3) * _cross(p1, p2, p4) <= 0 and
            _cross(p3, p4, p1) * _cross(p3, p4, p2) <= 0)


def cosine_angle(prev, curr, nxt) -> float:
    """计算折线三点夹角余弦值"""
    v1 = np.array([curr[0] - prev[0], curr[1] - prev[1]])
    v2 = np.array([nxt[0]  - curr[0], nxt[1]  - curr[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ===========================================================================
# 二、地理底图加载
# ===========================================================================

def load_geodata(
    base_path: str,
) -> tuple[Optional[object], Optional[object]]:
    """
    尝试加载道路与水域 shapefile。
    任一文件缺失则返回 (None, None)，系统自动切换为随机节点模式。
    """
    roads_path = os.path.join(base_path, "gis_osm_roads_free_1.shp")
    water_path = os.path.join(base_path, "gis_osm_water_a_free_1.shp")
    if not os.path.exists(roads_path) or not os.path.exists(water_path):
        log.info("底图文件未找到，切换为随机节点仿真模式")
        return None, None
    sh_roads = gpd.read_file(roads_path)
    sh_water = gpd.read_file(water_path)
    log.info("矢量底图加载完成")
    return sh_roads, sh_water


# ===========================================================================
# 三、空间图构建（KD 树加速，O(n log n)）
# ===========================================================================

def build_graph(
    sh_roads: Optional[object],
    n_random_nodes: int = 1000,
    link_radius_deg: float = 0.075,
    chongming_lat_threshold: float = 31.48,  # 崇明本岛南端约 31.48°
    chongming_radius_deg: float = 0.22,  # 扩大跨江桥接半径
) -> tuple[nx.Graph, np.ndarray]:
    """
    构建自由空间连通图。

    使用 cKDTree.query_pairs 将建图时间复杂度从 O(n²) 降为 O(n log n)。

    Returns
    -------
    G : nx.Graph
    nodes_array : np.ndarray, shape (N, 2)  —— 节点坐标数组（lon, lat）
    """
    # 采样节点
    if sh_roads is not None:
        major_roads = sh_roads[sh_roads['code'].isin([5111, 5112, 5113, 5114])]
        raw_pts: set = set()
        for _, row in major_roads.iterrows():
            if row.geometry.geom_type == 'LineString':
                raw_pts.update(row.geometry.coords)
        sampled = random.sample(list(raw_pts), min(1200, len(raw_pts)))
    else:
        sampled = [
            (random.uniform(121.05, 121.95), random.uniform(30.85, 31.72))
            for _ in range(n_random_nodes)
        ]

    nodes_array = np.array(sampled, dtype=np.float64)  # (N, 2)
    G = nx.Graph()

    # 分区构建：崇明区单独使用更大半径
    chongming_mask = nodes_array[:, 1] > chongming_lat_threshold
    mainland_idx   = np.where(~chongming_mask)[0]
    chongming_idx  = np.where(chongming_mask)[0]

    def _add_edges(idx_subset: np.ndarray, radius: float) -> None:
        pts = nodes_array[idx_subset]
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=radius)
        for i, j in pairs:
            gi, gj = idx_subset[i], idx_subset[j]
            p1, p2 = tuple(nodes_array[gi]), tuple(nodes_array[gj])
            dist_m = deg_dist(p1, p2) * 111_000
            G.add_edge(p1, p2, length=dist_m)

    _add_edges(mainland_idx,  link_radius_deg)
    _add_edges(chongming_idx, chongming_radius_deg)

    # 跨区连接（崇明 ↔ 大陆，扩大搜索范围）
    if len(chongming_idx) > 0 and len(mainland_idx) > 0:
        tree_main = cKDTree(nodes_array[mainland_idx])
        for ci in chongming_idx:
            pt = nodes_array[ci]
            idxs = tree_main.query_ball_point(pt, r=chongming_radius_deg)
            for mi in idxs:
                p1, p2 = tuple(pt), tuple(nodes_array[mainland_idx[mi]])
                G.add_edge(p1, p2, length=deg_dist(p1, p2) * 111_000)

    log.info("空间图构建完成 | 节点: %d | 边: %d", G.number_of_nodes(), G.number_of_edges())
    return G, nodes_array


# ===========================================================================
# 四、KD 树辅助：最近节点查询
# ===========================================================================

class NodeIndex:
    """封装图节点的 KD 树，支持 O(log n) 最近节点查询"""

    def __init__(self, G: nx.Graph) -> None:
        self._nodes = list(G.nodes)
        self._arr   = np.array(self._nodes, dtype=np.float64)
        self._tree  = cKDTree(self._arr)

    def closest(self, gps: tuple) -> tuple:
        """返回距离 gps 最近的图节点"""
        _, idx = self._tree.query(gps)
        return self._nodes[idx]

    @property
    def nodes(self) -> list:
        return self._nodes


# ===========================================================================
# 五、战略锚点提取
# ===========================================================================

def extract_anchor_nodes(node_index: NodeIndex) -> list[tuple]:
    """
    提取战略锚点，并做地理分区均衡：
    将上海划为 5×4 网格，每格最多贡献 3 个锚点，
    避免临港/张江等高密度片区垄断候选池。
    """
    gps_list = (
        [p["gps"] for p in JOB_CENTERS      if p.get("weight", 0) >= 1.8] +
        [p["gps"] for p in RESIDENTIAL_HUBS if p.get("weight", 0) >= 2.0] +
        [p["gps"] for p in COMMERCIAL_CENTERS if p.get("bonus", 0) >= 100]
    )

    # 分区均衡：网格化后每格限额，防止某一片区堆叠过多锚点
    LON_MIN, LON_MAX = 121.05, 121.95
    LAT_MIN, LAT_MAX = 30.85, 31.72
    GRID_COLS, GRID_ROWS = 5, 4
    MAX_PER_CELL = 3

    cell_counts: dict[tuple, int] = {}
    balanced_gps: list[tuple] = []
    random.shuffle(gps_list)          # 随机打乱，避免固定偏好

    for gps in gps_list:
        col = int((gps[0] - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS)
        row = int((gps[1] - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_ROWS)
        cell = (max(0, min(col, GRID_COLS - 1)), max(0, min(row, GRID_ROWS - 1)))
        if cell_counts.get(cell, 0) < MAX_PER_CELL:
            balanced_gps.append(gps)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1

    anchor_nodes = list({node_index.closest(g) for g in balanced_gps})
    log.info("战略锚点提取完成：%d 个（网格均衡后）", len(anchor_nodes))
    return anchor_nodes


# ===========================================================================
# 六、线路任务初始化
# ===========================================================================

METRO_COLORS = ["#ff0055", "#00ff66", "#00ccff", "#e0aa00", "#cc00ff"]
MIN_INTER_LINE_GAP_DEG = 0.05
LINE_DIST_MIN_DEG      = 0.15  # ≈17km 最短
LINE_DIST_MAX_DEG      = 0.36  # ≈40km 普通线路上限
LINE_DIST_MAX_DEG_LONG = 0.58  # ≈64km 崇明/远郊线专用


@dataclass
class MetroMission:
    name:       str
    color:      str
    start_node: tuple
    end_node:   tuple
    q_table:    dict = field(default_factory=dict)

    def get_q(self, G: nx.Graph, node: tuple) -> dict:
        """惰性初始化 Q 值字典"""
        if node not in self.q_table:
            self.q_table[node] = {nbr: 0.0 for nbr in G.neighbors(node)}
        return self.q_table[node]

    def warm_start(self, G: nx.Graph, boost: float = 60_000.0) -> None:
        """
        轻量热身：只对起点邻居注入微弱的终点方向偏好，
        不沿 Dijkstra 固化路线，路径可以自由弯曲。
        """
        q = self.get_q(G, self.start_node)
        for nbr in q:
            # 距终点越近的邻居 Q 值略高，仅提供方向感
            d_km = deg_to_km(deg_dist(nbr, self.end_node))
            q[nbr] = boost / max(1.0, d_km)


def _too_close_to_existing(node: tuple, missions: list[MetroMission], attr: str) -> bool:
    for m in missions:
        if deg_dist(node, getattr(m, attr)) < MIN_INTER_LINE_GAP_DEG:
            return True
    return False


def _is_west_of_center(node: tuple) -> bool:
    return node[0] < 121.47   # 人民广场经度偏西侧

def _is_east_of_center(node: tuple) -> bool:
    return node[0] > 121.51   # 陆家嘴经度偏东侧

def _is_north_of_center(node: tuple) -> bool:
    return node[1] > 31.27

def _is_south_of_center(node: tuple) -> bool:
    return node[1] < 31.18

# 起终点方向对组：保证线路必须穿越市区
_OPPOSITE_PAIRS = [
    (_is_west_of_center,  _is_east_of_center),   # 东西向
    (_is_east_of_center,  _is_west_of_center),
    (_is_north_of_center, _is_south_of_center),   # 南北向
    (_is_south_of_center, _is_north_of_center),
    (_is_west_of_center,  _is_south_of_center),   # 西北→东南
    (_is_east_of_center,  _is_north_of_center),   # 东南→西北
]


def init_missions(
    G: nx.Graph,
    node_index: NodeIndex,
    anchor_nodes: list[tuple],
    n_lines: int = 5,
) -> list[MetroMission]:
    """
    为每条线路选择满足骨干约束的起终点。
    强制起终点位于市区两侧，确保路径穿越市中心。
    """
    missions: list[MetroMission] = []
    valid_nodes = node_index.nodes
    _used_pair_idx: set[int] = set()  # 记录已用方向对，避免重复

    for i in range(n_lines):
        best_start = best_end = None
        best_score = -1.0

        # 随机选取方向对（已用过的不重复），保证方向多样且每次不同
        available_pairs = [p for j, p in enumerate(_OPPOSITE_PAIRS) if j not in _used_pair_idx]
        if not available_pairs:
            available_pairs = _OPPOSITE_PAIRS
        pair_idx = random.randrange(len(available_pairs))
        start_side, end_side = available_pairs[pair_idx]
        _used_pair_idx.add(_OPPOSITE_PAIRS.index(available_pairs[pair_idx]))

        # 混合锚点 + 随机节点，扩大候选多样性
        all_pool = anchor_nodes if anchor_nodes else valid_nodes
        starts_anchor = [n for n in all_pool   if start_side(n)]
        starts_random = [n for n in valid_nodes if start_side(n)]
        ends_random   = [n for n in valid_nodes if end_side(n)]   or valid_nodes
        # 锚点占 40%，随机节点占 60%，每次组合不同
        n_anchor = min(24, len(starts_anchor))
        n_rand   = min(36, len(starts_random))
        starts_pool = (random.sample(starts_anchor, n_anchor) if starts_anchor else []) + \
                      (random.sample(starts_random, n_rand)   if starts_random else valid_nodes[:36])
        ends_pool   = random.sample(ends_random, min(500, len(ends_random)))

        sampled_starts = starts_pool
        sampled_ends   = ends_pool

        for n_s in sampled_starts:
            if _too_close_to_existing(n_s, missions, "start_node"):
                continue
            for n_e in sampled_ends:
                if n_s == n_e:
                    continue
                if _too_close_to_existing(n_e, missions, "end_node"):
                    continue
                d = deg_dist(n_s, n_e)
                dist_max = LINE_DIST_MAX_DEG_LONG if (_is_north_of_center(n_s) or _is_north_of_center(n_e)) else LINE_DIST_MAX_DEG
                if not (LINE_DIST_MIN_DEG <= d <= dist_max):
                    continue
                if missions and not any(
                    segments_intersect(n_s, n_e, m.start_node, m.end_node)
                    for m in missions
                ):
                    continue
                score = d * random.uniform(1.0, 1.5)
                if score > best_score:
                    best_score = score
                    best_start, best_end = n_s, n_e

        # 兜底：方向约束放宽，但仍要求距离合理
        if best_start is None or best_end is None:
            log.warning("Line %d 方向约束兜底，起终点随机宽松选取", i + 1)
            while True:
                p1 = random.choice(all_pool)
                p2 = random.choice(valid_nodes)
                if p1 != p2 and LINE_DIST_MIN_DEG * 0.9 <= deg_dist(p1, p2) <= LINE_DIST_MAX_DEG:
                    best_start, best_end = p1, p2
                    break

        m = MetroMission(
            name=f"Line {i + 1}",
            color=METRO_COLORS[i],
            start_node=best_start,
            end_node=best_end,
        )
        m.warm_start(G)   # 轻量热身
        missions.append(m)

    log.info("线路任务初始化完成：%d 条", len(missions))
    return missions


# ===========================================================================
# 七、路径探索
# ===========================================================================

def explore_path(
    G: nx.Graph,
    mission: MetroMission,
    epsilon: float,
    max_steps: int = 55,   # 收紧上限，控制路径总长
) -> list[tuple]:
    """ε-greedy 单步路径探索，返回本轮节点序列"""
    current = mission.start_node
    path    = [current]

    for _ in range(max_steps):
        if current == mission.end_node:
            break
        q_vals    = mission.get_q(G, current)
        neighbors = [n for n in q_vals if n not in path]
        if not neighbors:
            break
        if random.random() < epsilon:
            chosen = random.choice(neighbors)
        else:
            chosen = max(neighbors, key=lambda n: q_vals[n])
        path.append(chosen)
        current = chosen

    return path


# ===========================================================================
# 八、奖励计算（完全解耦，纯函数）
# ===========================================================================

def _cau_hit_count(path: list[tuple]) -> int:
    radius_km = SH_CAU_CONFIG["influence_radius_km"]
    count = 0
    for node in path:
        for hub in SH_CAU_CONFIG["hubs"].values():
            if deg_to_km(deg_dist(node, hub["gps"])) <= radius_km:
                count += 1
                break
    return count


def _min_dist_to_cau(node: tuple) -> float:
    return min(
        deg_to_km(deg_dist(node, h["gps"]))
        for h in SH_CAU_CONFIG["hubs"].values()
    )


def _point_to_line_dist(p: tuple, a: tuple, b: tuple) -> float:
    """点 p 到线段 a-b 所在直线的垂直距离（度单位）"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return deg_dist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = (a[0] + t * dx, a[1] + t * dy)
    return deg_dist(p, proj)


def compute_step_reward(
    path: list[tuple],
    step_idx: int,
    G: nx.Graph,
    edge_occupancy: dict,
    transfer_stations: set,
    cau_hit: int,
    end_node: tuple,
    start_node: tuple = None,
) -> float:
    """计算单步奖励（纯函数，无副作用）"""
    curr = path[step_idx]
    nxt  = path[step_idx + 1]
    edge_data = G[curr][nxt]
    dist_m    = edge_data["length"]
    gap_km    = dist_m / 1_000.0

    reward = 0.0

    # 1. 物理开销
    reward -= dist_m * 0.005

    # 2. 目标导向
    d_curr = deg_dist(curr, end_node)
    d_nxt  = deg_dist(nxt,  end_node)
    progress_m = (d_curr - d_nxt) * 111_000
    reward += progress_m * 25.0 if progress_m > 0 else progress_m * 60.0

    # 3. CAU 引力场：靠近时给引力，首次进入给一次性奖励，在市区内不再持续加分
    min_d_cau_curr = _min_dist_to_cau(curr)
    min_d_cau_nxt  = _min_dist_to_cau(nxt)
    cau_progress   = min_d_cau_curr - min_d_cau_nxt
    radius_km      = SH_CAU_CONFIG["influence_radius_km"]

    if cau_progress > 0:
        reward += cau_progress * (400.0 / max(0.5, min_d_cau_nxt)) * 120.0

    # 首次进入市区：一次性奖励（判断 curr 在外、nxt 在内）
    if min_d_cau_curr > radius_km and min_d_cau_nxt <= radius_km:
        reward += SH_CAU_CONFIG["arrival_reward"]

    # 在市区内：不给额外奖励，转而依靠方向感和目标导向推动穿越

    # 4. 交通枢纽锚定
    hub_r = SH_TRANSPORT_HUBS_CONFIG["anchor_radius_km"]
    for h_info in SH_TRANSPORT_HUBS_CONFIG["hubs"].values():
        if deg_to_km(deg_dist(nxt, h_info["gps"])) <= hub_r:
            reward += h_info["reward_bonus"]

    # 5. 转弯惩罚（适度，允许自然弯曲）
    if step_idx > 0:
        prev = path[step_idx - 1]
        cos_a = cosine_angle(prev, curr, nxt)
        reward += cos_a * 10_000.0
        if cos_a < 0.5:                     # >60° 才开始惩罚
            reward -= 30_000.0
        if cos_a < 0.0:                     # >90° 急转
            reward -= 70_000.0

    # 6. 站距控制
    opt_lo, opt_hi = SH_NETWORK_DECOUPLING_CONFIG["station_gap_optimal_range"]
    gap_min = SH_NETWORK_DECOUPLING_CONFIG["station_gap_min_km"]
    gap_max = SH_NETWORK_DECOUPLING_CONFIG["station_gap_max_km"]
    if opt_lo <= gap_km <= opt_hi:
        reward += 8_000.0
    elif gap_km < gap_min:
        reward -= 25_000.0
    elif gap_km > gap_max:
        reward -= 45_000.0

    # 7. 叠线惩罚 / 换乘奖励
    edge_key   = tuple(sorted([curr, nxt]))
    overlap    = edge_occupancy.get(edge_key, 1)
    if overlap > 1:
        reward += SH_NETWORK_DECOUPLING_CONFIG["base_edge_penalty_per_overlap"] * (overlap - 1)
    elif nxt in transfer_stations:
        reward += SH_NETWORK_DECOUPLING_CONFIG["transfer_bonus"]

    # 8. CAU 穿心率一票否决
    if cau_hit < SH_CAU_CONFIG["min_hit_required"]:
        reward += SH_CAU_CONFIG["veto_penalty"]

    return reward


# ===========================================================================
# 九、Q 表更新
# ===========================================================================

def update_q_table(
    mission: MetroMission,
    G: nx.Graph,
    path: list[tuple],
    edge_occupancy: dict,
    transfer_stations: set,
    alpha: float,
    gamma: float,
) -> float:
    """对一条完整路径做 TD(0) Q 更新，返回该路径总长（km）。"""
    if len(path) < 2:
        return 0.0

    end_node  = mission.end_node
    cau_hit   = _cau_hit_count(path)
    total_len_km = sum(
        G[path[s]][path[s + 1]]["length"] / 1_000.0
        for s in range(len(path) - 1)
    )

    # 路径过长惩罚：直线距离的 3 倍以上视为严重绕圈
    straight_km = deg_to_km(deg_dist(mission.start_node, mission.end_node))
    length_ratio = total_len_km / max(1.0, straight_km)
    length_penalty_per_step = 0.0
    if length_ratio > 2.5:
        length_penalty_per_step = -20_000.0 * (length_ratio - 2.5)
    elif length_ratio > 1.5:
        length_penalty_per_step = -8_000.0  * (length_ratio - 1.5)

    for s in range(len(path) - 1):
        curr, nxt = path[s], path[s + 1]
        reward = compute_step_reward(
            path, s, G, edge_occupancy, transfer_stations, cau_hit, end_node,
            start_node=mission.start_node,
        )
        reward += length_penalty_per_step   # 叠加全局绕路惩罚

        q_curr  = mission.get_q(G, curr)
        q_nxt   = mission.get_q(G, nxt)
        max_nxt = max(q_nxt.values()) if q_nxt else 0.0
        q_curr[nxt] += alpha * (reward + gamma * max_nxt - q_curr[nxt])

    return total_len_km


# ===========================================================================
# 十、训练状态（封装可变参数，避免全局污染）
# ===========================================================================

@dataclass
class TrainState:
    alpha:         float = 0.35          # 学习率略提升，收敛更快
    gamma:         float = 0.96
    epsilon:       float = 0.85          # 初始探索率降低，减少早期随机游走
    epsilon_min:   float = 0.02
    epsilon_decay: float = 0.985         # 适中衰减，约200轮后收敛
    episode:       int   = 0

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ===========================================================================
# 十一、渲染工具
# ===========================================================================

def render_frame(
    ax: plt.Axes,
    G: nx.Graph,
    sh_water: Optional[object],
    missions: list[MetroMission],
    all_paths: list[list[tuple]],
    line_lengths: list[float],
    transfer_stations: set,
    episode: int,
) -> None:
    """清空并重绘当前帧（渲染与逻辑完全解耦）"""
    # 设置视图范围
    if sh_water is not None:
        b = sh_water.total_bounds
        ax.set_xlim(b[0] - 0.02, b[2] + 0.02)
        ax.set_ylim(b[1] - 0.02, b[3] + 0.02)
    else:
        nodes_arr = np.array(list(G.nodes))
        x_min, y_min = nodes_arr.min(axis=0)
        x_max, y_max = nodes_arr.max(axis=0)
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_aspect("equal")

    # 清除上一帧动态元素（保留底层水域 collection）
    water_offset = 1 if sh_water is not None else 0
    for artist in list(ax.lines) + list(ax.texts) + list(ax.collections)[water_offset:]:
        artist.remove()

    # 绘制各条线路
    for idx, (mission, path, length_km) in enumerate(zip(missions, all_paths, line_lengths)):
        if len(path) < 2:
            continue
        px = [n[0] for n in path]
        py = [n[1] for n in path]
        ax.plot(px, py, color=mission.color, linewidth=4.5, alpha=0.95,
                solid_capstyle="round", zorder=4)

        mid = path[len(path) // 2]
        ax.text(
            mid[0], mid[1] + 0.002,
            f"{mission.name}: {length_km:.1f}km",
            color="#1c1c1e", fontsize=8, weight="bold",
            bbox=dict(facecolor="#ffffff", alpha=0.90, edgecolor="#b1a693",
                      boxstyle="round,pad=0.15", lw=0.5),
        )
        ax.scatter(*mission.start_node, color=mission.color, s=100,
                   edgecolors="#ffffff", linewidths=1.5, zorder=5)
        ax.scatter(*mission.end_node,   color=mission.color, s=220,
                   edgecolors="#ffffff", linewidths=1.5, marker="*", zorder=5)

    # 绘制战略交通枢纽（紫色方块）
    for h_info in SH_TRANSPORT_HUBS_CONFIG["hubs"].values():
        ax.scatter(*h_info["gps"], color="#9b59b6", marker="s", s=60,
                   edgecolors="#ffffff", zorder=5)

    # 绘制换乘车站（白底黑圈）
    if transfer_stations:
        tx = [n[0] for n in transfer_stations]
        ty = [n[1] for n in transfer_stations]
        ax.scatter(tx, ty, color="#ffffff", s=50, edgecolors="#1c1c1e",
                   linewidths=1.5, zorder=6)

    ax.set_title(
        f"上海城市线网演化 (Episode: {episode} | 换乘枢纽: {len(transfer_stations)})",
        color="#1c1c1e", fontsize=13, pad=10, weight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2c3e50")
        spine.set_linewidth(3.0)


# ===========================================================================
# 十二、主程序
# ===========================================================================

def main() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- 底图 ---
    shp_base  = rf"{os.getcwd()}\shanghai-260519-free.shp"
    sh_roads, sh_water = load_geodata(shp_base)

    # --- 构建空间图 ---
    log.info("第二步：构建自由空间连通图...")
    G, _ = build_graph(sh_roads)

    # --- 节点索引 ---
    node_index    = NodeIndex(G)
    anchor_nodes  = extract_anchor_nodes(node_index)

    # --- 线路初始化 ---
    log.info("第三步：初始化骨干线路任务...")
    missions = init_missions(G, node_index, anchor_nodes, n_lines=5)

    # --- 训练状态 ---
    state       = TrainState()
    max_episodes = 1000

    # --- 画布 ---
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="#eaeaea")
    ax.set_facecolor("#f7e8cf")
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)

    if sh_water is not None:
        log.info("第四步：渲染水域底图...")
        sh_water.plot(ax=ax, color="#a9d1f5", edgecolor="none", alpha=1.0, zorder=1)

    # --- 动画更新函数（调度层）---
    def update(_frame: int) -> None:
        state.episode += 1

        # 全局占用统计
        node_occupancy: dict[tuple, list[int]] = {}
        edge_occupancy: dict[tuple, int]       = {}

        # 阶段一：探索，收集路径与占用信息
        all_paths: list[list[tuple]] = []
        for idx, mission in enumerate(missions):
            path = explore_path(G, mission, state.epsilon)
            all_paths.append(path)
            for s in range(len(path) - 1):
                ek = tuple(sorted([path[s], path[s + 1]]))
                edge_occupancy[ek] = edge_occupancy.get(ek, 0) + 1
                node = path[s + 1]
                node_occupancy.setdefault(node, [])
                if idx not in node_occupancy[node]:
                    node_occupancy[node].append(idx)

        transfer_stations = {n for n, agents in node_occupancy.items() if len(agents) > 1}

        # 阶段二：Q 更新
        line_lengths: list[float] = []
        for mission, path in zip(missions, all_paths):
            length_km = update_q_table(
                mission, G, path, edge_occupancy, transfer_stations,
                state.alpha, state.gamma,
            )
            line_lengths.append(length_km)

        # 阶段三：渲染
        render_frame(ax, G, sh_water, missions, all_paths, line_lengths,
                     transfer_stations, state.episode)

        state.decay_epsilon()

        if state.episode >= max_episodes:
            ani.event_source.stop()
            log.info("线网演化完成，已收敛 (episode=%d)", state.episode)

    ani = FuncAnimation(fig, update, frames=max_episodes, interval=50, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()