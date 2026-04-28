#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum (E7.0).
清扫大作战特征预处理器（E7.0）。

E7.0 = E5.2 base (global_map, charger_cells, BFS support)
      + E6.0 separated critic (16D global features)
      + new reward shaping (exploration, idle penalty, NPC death)
"""
from collections import deque

import numpy as np


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum (E7.0).

    清扫大作战特征预处理器（E7.0）。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径
    LOW_BATTERY_THRESHOLD = 0.35

    # E7.0 reward constants / E7.0 奖励常数
    REVISIT_PENALTY = -0.03  # Stronger revisit penalty to discourage local loops
    CLEANING_EFFICIENCY_BONUS_COEF = 0.05  # Dense shaping from cleaned-per-step efficiency
    HIGH_BATTERY_NEAR_CHARGER_PENALTY = -0.3
    POST_CHARGE_LINGER_PENALTY = -0.2
    CHARGER_STAGNATION_PENALTY = -0.1
    CHARGER_LINGER_RADIUS = 4
    FIRST_LIFE_CLEAN_REWARD = 0.008
    CHARGER_POSITION_REWARD_COEF = 0.02
    STREAK_THRESHOLD = 3
    STREAK_BONUS = 0.01
    EXPLORATION_BONUS = 0.05
    IDLE_PENALTY = -0.15          # E7.2: strengthened from -0.05
    STAGNATION_PENALTY = -0.05    # E7.2: no new cell cleaned in last 5 steps
    STAGNATION_STEPS = 5

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        # Global memory map (E5.2): 0=unknown, 1=obstacle, 2=clean, 3=dirt
        # 全局记忆地图：0=未知, 1=障碍, 2=已清扫, 3=污渍
        self.global_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.charger_cells = set()
        self.chargers_initialized = False

        self.step_no = 0
        self.prev_step_no = 0
        self.battery = 600
        self.battery_max = 600
        self.prev_battery = 600

        self.cur_pos = (0, 0)
        self.prev_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1
        self.consecutive_clean_count = 0

        # Global passable map (0=obstacle, 1=passable), used for BFS and ray
        # 全局通行地图（0=障碍, 1=可通行），用于 BFS 和射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance / 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8
        self._chargers = []
        self._npcs = []
        self.terminated = False

        # E7.0 exploration state / E7.0 探索状态
        self.visited = set()
        self.last_pos = None
        self.steps_since_new_cell = 0  # E7.2: stagnation counter
        self.recent_positions = deque(maxlen=6)
        self.phase_charge_mode = False
        self.phase_post_charge_steps_left = 0
        self.phase_safety_margin = 0
        self.has_charged = False
        self.prev_two_charger_metric = None

    # ------------------------------------------------------------------
    # Charger helpers (E5.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _charger_center(charger):
        """Return charger center in grid coordinates.

        返回充电桩中心点栅格坐标。
        """
        pos = charger.get("pos", {})
        cx = int(pos.get("x", 0)) + int(charger.get("w", 1)) // 2
        cz = int(pos.get("z", 0)) + int(charger.get("h", 1)) // 2
        return cx, cz

    def _nearest_charger(self, pos):
        """Return nearest charger Chebyshev distance and direction.

        返回最近充电桩的切比雪夫距离及方向。
        """
        if not self._chargers:
            return float(self.GRID_SIZE), 0.0, 0.0

        hx, hz = pos
        min_dist = float(self.GRID_SIZE)
        best_dx = 0.0
        best_dz = 0.0

        for charger in self._chargers:
            cx, cz = self._charger_center(charger)
            dx = float(cx - hx)
            dz = float(cz - hz)
            dist = float(max(abs(dx), abs(dz)))
            if dist < min_dist:
                min_dist = dist
                best_dx = dx
                best_dz = dz

        return min_dist, best_dx, best_dz

    def _nearest_reachable_chargers(self):
        from agent_ppo.pathfinding import reachable_charger_distances

        return reachable_charger_distances(self, self.cur_pos[0], self.cur_pos[1], self._npcs)

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.prev_step_no = self.step_no
        self.step_no = int(observation["step_no"])
        self.prev_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.prev_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)
        if self.step_no > 1 and self.battery > self.prev_battery + 1:
            self.has_charged = True

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]
        self._npcs = list(frame_state.get("npcs") or [])
        self._chargers = list(frame_state.get("organs") or [])
        self.terminated = bool(env_obs.get("terminated", False))

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            self.update_global_map(self.cur_pos, map_info, self._chargers)

    # ------------------------------------------------------------------
    # Global map maintenance (E5.2)
    # ------------------------------------------------------------------

    def update_global_map(self, hero_pos, map_info, organs):
        """Update the global memory map from the current 21x21 observation.

        用当前 21x21 观测更新全局记忆地图。
        """
        hx, hz = hero_pos

        if not self.chargers_initialized and organs:
            for organ in organs:
                pos = organ.get("pos", {})
                width = int(organ.get("w", 1))
                height = int(organ.get("h", 1))
                base_x = int(pos.get("x", 0))
                base_z = int(pos.get("z", 0))
                for dx in range(width):
                    for dz in range(height):
                        self.charger_cells.add((base_x + dx, base_z + dz))
            self.chargers_initialized = True

        for i in range(21):
            for j in range(21):
                gx = hx - self.VIEW_HALF + j
                gz = hz - self.VIEW_HALF + i
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue

                cell = int(map_info[i][j])
                if cell == 0:
                    self.global_map[gz, gx] = 1
                elif cell == 1:
                    self.global_map[gz, gx] = 2
                elif cell == 2:
                    self.global_map[gz, gx] = 3

                self.passable_map[gz, gx] = 1 if cell != 0 else 0

    def is_passable(self, x, z, allow_unknown=True):
        """Return whether the cell is traversable under the requested certainty."""
        if x < 0 or x >= self.GRID_SIZE or z < 0 or z >= self.GRID_SIZE:
            return False
        cell = int(self.global_map[z, x])
        if cell == 1:
            return False
        if allow_unknown or cell != 0:
            return True
        return (x, z) in self.charger_cells

    def is_known_passable(self, x, z):
        """Return whether the cell is a confirmed traversable cell."""
        return self.is_passable(x, z, allow_unknown=False)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (19D).

        全局状态特征（19D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
          [12] low_battery       phase feature / 低电量阶段标记
          [13] returning_charge  phase feature / 回桩阶段标记
          [14] just_charged      phase feature / 刚充满阶段标记
          [15] return_budget     battery - bfs_dist / 回桩剩余预算
          [16] return_margin     battery - (bfs_dist + safety) / 回桩安全余量
          [17] new_cell_gap      steps since new cell / 距离上次新格子的步数
          [18] local_cycle_ratio repeated-position ratio / 近期局部循环比例
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        low_battery = 1.0 if battery_ratio < self.LOW_BATTERY_THRESHOLD else 0.0
        returning_charge = 1.0 if self.phase_charge_mode else 0.0
        just_charged = 1.0 if self.phase_post_charge_steps_left > 0 else 0.0
        return_budget, return_margin = self._get_return_budget_features()
        new_cell_gap_norm = _norm(self.steps_since_new_cell, 10)
        local_cycle_ratio = self._calc_local_cycle_ratio()

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
                low_battery,
                returning_charge,
                just_charged,
                return_budget,
                return_margin,
                new_cell_gap_norm,
                local_cycle_ratio,
            ],
            dtype=np.float32,
        )

    def _get_return_budget_features(self):
        from agent_ppo.pathfinding import bfs_to_charger

        bfs_dist, bfs_action = bfs_to_charger(self, self.cur_pos[0], self.cur_pos[1], self._npcs)
        if bfs_action == -1 and bfs_dist != 0:
            bfs_dist = float(self.GRID_SIZE)

        denom = max(float(self.battery_max), 1.0)
        return_budget = float(np.clip((self.battery - bfs_dist) / denom, -1.0, 1.0))
        safety_margin = float(self.phase_safety_margin)
        return_margin = float(np.clip((self.battery - (bfs_dist + safety_margin)) / denom, -1.0, 1.0))
        return return_budget, return_margin

    def _calc_local_cycle_ratio(self):
        if not self.recent_positions:
            return 0.0
        unique = len(set(self.recent_positions))
        return float(1.0 - unique / max(len(self.recent_positions), 1))

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def _get_charger_feature(self):
        """Build the 12D charger/global anchor feature."""
        reachable = self._nearest_reachable_chargers()
        battery_ratio = _norm(self.battery, self.battery_max)
        low_battery_flag = 1.0 if battery_ratio < self.LOW_BATTERY_THRESHOLD else 0.0
        reachable_count = len(reachable)
        bfs_valid = 1.0 if reachable_count > 0 else 0.0

        def charger_slot(idx):
            if idx < reachable_count:
                dist, (cx, cz) = reachable[idx]
                dx = float(cx - self.cur_pos[0]) / self.GRID_SIZE
                dz = float(cz - self.cur_pos[1]) / self.GRID_SIZE
                return [_norm(dist, self.GRID_SIZE), dx, dz]
            return [1.0, 0.0, 0.0]

        slot1 = charger_slot(0)
        slot2 = charger_slot(1)
        nearest_two_mean_dist = float(sum(d for d, _ in reachable[:2]) / max(len(reachable[:2]), 1)) if reachable else float(self.GRID_SIZE)
        nearest_two_mean_dist_norm = _norm(nearest_two_mean_dist, self.GRID_SIZE)
        charger_centrality = 1.0 - nearest_two_mean_dist_norm
        reachable_count_norm = min(reachable_count, 4) / 4.0

        return np.array(
            slot1
            + slot2
            + [
                reachable_count_norm,
                bfs_valid,
                nearest_two_mean_dist_norm,
                charger_centrality,
                battery_ratio,
                low_battery_flag,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Critic global features (E6.0)
    # ------------------------------------------------------------------

    def _extract_critic_global_features(self):
        """Build critic-only global features (19D).

        16D: 4 charger centers + 4 NPC positions (coords /128)
        3D:  explicit progress signals (E7.4)
          - cleaning_ratio: dirt_cleaned / total_dirt
          - battery_ratio:  battery / battery_max
          - step_ratio:     step_no / max_step
        Critic 专用 19D 特征：16D 坐标 + 3D 进度信号（E7.4 新增）。
        """
        critic_global = []

        for idx in range(4):
            if idx < len(self._chargers):
                charger = self._chargers[idx]
                pos = charger.get("pos", {})
                cx = (float(pos.get("x", 0)) + float(charger.get("w", 1)) / 2.0) / self.GRID_SIZE
                cz = (float(pos.get("z", 0)) + float(charger.get("h", 1)) / 2.0) / self.GRID_SIZE
                critic_global.extend([cx, cz])
            else:
                critic_global.extend([-1.0, -1.0])

        for idx in range(4):
            if idx < len(self._npcs):
                npc = self._npcs[idx]
                pos = npc.get("pos", {})
                nx = float(pos.get("x", 0)) / self.GRID_SIZE
                nz = float(pos.get("z", 0)) / self.GRID_SIZE
                critic_global.extend([nx, nz])
            else:
                critic_global.extend([-1.0, -1.0])

        # E7.4: explicit progress signals
        cleaning_ratio = self.dirt_cleaned / max(self.total_dirt, 1)
        battery_ratio = self.battery / max(self.battery_max, 1)
        step_ratio = self.step_no / 1000.0
        critic_global.extend([cleaning_ratio, battery_ratio, step_ratio])

        return np.array(critic_global, dtype=np.float32)

    # ------------------------------------------------------------------
    # Main feature process
    # ------------------------------------------------------------------

    def feature_process(self, env_obs, last_action):
        """Generate 81D feature, legal action mask, reward, and 19D critic extra.

        生成 81D 特征向量、合法动作掩码、标量奖励和 19D Critic 额外特征。
        """
        self.pb2struct(env_obs, last_action)
        self.recent_positions.append(self.cur_pos)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 19D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)
        charger_feature = self._get_charger_feature()  # 5D

        feature = np.concatenate([local_view, global_state, legal_arr, charger_feature])  # 81D

        reward = self.reward_process()
        critic_extra = self._extract_critic_global_features()  # 16D

        return feature, legal_action, reward, critic_extra

    # ------------------------------------------------------------------
    # Reward shaping (E7.0)
    # ------------------------------------------------------------------

    def reward_process(self):
        """E7.0 reward: RL learns cleaning + exploration only.

        E7.0 奖励：RL 只学清扫和探索，充电由 BFS 接管，NPC 由安全层接管。
        """
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_efficiency = self.dirt_cleaned / max(self.step_no, 1)
        first_life = not self.has_charged
        reward = 0.0

        # ========== Core cleaning / 清扫核心 ==========

        # 1) Cleaning reward (unchanged) / 清扫奖励（不变）
        reward += (self.FIRST_LIFE_CLEAN_REWARD if first_life else 0.1) * cleaned_this_step

        # 1.2) First-life charger positioning reward / 第一条命的充电桩站位奖励
        if first_life:
            current_metric = self._nearest_two_reachable_charger_metric()
            if self.prev_two_charger_metric is not None and current_metric is not None:
                reward += self.CHARGER_POSITION_REWARD_COEF * (self.prev_two_charger_metric - current_metric)
            self.prev_two_charger_metric = current_metric
        else:
            self.prev_two_charger_metric = None

        # 1.5) Dense reward from cleaned-per-step efficiency / 单步清扫效率奖励
        reward += self.CLEANING_EFFICIENCY_BONUS_COEF * cleaning_efficiency

        # 2) Step penalty (unchanged) / 步数惩罚（不变）
        reward -= 0.001

        # 2.5) Discourage lingering near chargers with sufficient battery.
        battery_ratio = _norm(self.battery, self.battery_max)
        nearest_charger_dist, _, _ = self._nearest_charger(self.cur_pos)
        if not self.phase_charge_mode and cleaned_this_step == 0 and battery_ratio > 0.75 and nearest_charger_dist <= 3:
            reward += self.HIGH_BATTERY_NEAR_CHARGER_PENALTY

        if not self.phase_charge_mode and cleaned_this_step == 0 and nearest_charger_dist <= self.CHARGER_LINGER_RADIUS:
            if self.phase_post_charge_steps_left > 0 and self.steps_since_new_cell >= 2:
                reward += self.POST_CHARGE_LINGER_PENALTY
            elif self.steps_since_new_cell >= self.STAGNATION_STEPS:
                reward += self.CHARGER_STAGNATION_PENALTY

        # 3) Revisit penalty (E2.1 sweet spot) / 重复访问惩罚
        if cleaned_this_step == 0:
            reward += self.REVISIT_PENALTY
            self.consecutive_clean_count = 0
        else:
            self.consecutive_clean_count += 1
            if self.consecutive_clean_count >= self.STREAK_THRESHOLD:
                reward += self.STREAK_BONUS

        # ========== Exploration (E7.0 new) / 探索（E7.0 新增）==========

        # 5) First-visit bonus / 首次访问奖励
        hero_pos = self.cur_pos
        if hero_pos not in self.visited:
            self.visited.add(hero_pos)
            reward += self.EXPLORATION_BONUS
            self.steps_since_new_cell = 0
        else:
            self.steps_since_new_cell += 1

        # 6) Idle penalty / 原地不动惩罚（E7.2 strengthened）
        if self.last_pos is not None and hero_pos == self.last_pos:
            reward += self.IDLE_PENALTY
        self.last_pos = hero_pos

        # 7) Stagnation penalty / 无进展惩罚（E7.2 new）
        if self.steps_since_new_cell >= self.STAGNATION_STEPS:
            reward += self.STAGNATION_PENALTY

        # ========== Death penalties / 死亡惩罚 ==========

        # 8) Battery depletion death / 电量耗尽死亡
        if self.terminated and self.battery <= 0:
            reward -= 5.0

        # 9) NPC collision death / NPC 碰撞死亡
        if self.terminated and self.battery > 0:
            reward -= 5.0

        # ========== Removed (E7.0) / 已删除 ==========
        # ❌ charge_bonus (BFS handles charging)
        # ❌ approach_reward (BFS handles charging)
        # ❌ npc_proximity_penalty (safety layer handles NPC avoidance)

        return reward

    def _nearest_two_reachable_charger_metric(self):
        from agent_ppo.pathfinding import reachable_charger_distances

        results = reachable_charger_distances(self, self.cur_pos[0], self.cur_pos[1], self._npcs)
        if not results:
            return None

        top2 = [dist for dist, _ in results[:2]]
        return float(sum(top2) / len(top2))
