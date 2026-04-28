#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent (E7.0).
清扫大作战 Agent 主类（E7.0）。

E7.0 = E5.2 BFS charging + E6.0 separated Actor/Critic + E7.0 NPC safety layer
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from agent_ppo.pathfinding import bfs_to_charger, npc_safe_filter
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    EPISODE_MAX_STEP = 1000
    SAFETY_MARGIN = 60  # Battery buffer above BFS path length for charge trigger
    DIRECT_CHARGE_CHEB_MULTIPLIER = 2.0
    DIRECT_CHARGE_MARGIN = 25
    JUST_FINISHED_CHARGE_STEPS = 20
    CHARGE_TRIGGER_CONFIRM_STEPS = 2
    CHARGE_CRITICAL_BUFFER = 20
    CHARGE_RELEASE_BUFFER = 10
    CHARGE_LOW_BATTERY_RATIO = 0.3
    CHARGE_URGENT_NPC_BLOCK_RADIUS = 2
    CHARGE_URGENT_SAFE_DIST = 2

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": Config.INIT_LEARNING_RATE_START},
                {"params": self.model.critic.parameters(), "lr": Config.INIT_LEARNING_RATE_START * 3},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.charge_mode = False
        self.charge_trigger_count = 0
        self.just_finished_charge_steps_left = 0
        self.last_action = -1
        self.last_reward = 0.0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state."""
        self.preprocessor = Preprocessor()
        self.charge_mode = False
        self.charge_trigger_count = 0
        self.just_finished_charge_steps_left = 0
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData."""
        self.preprocessor.phase_charge_mode = self.charge_mode
        self.preprocessor.phase_post_charge_steps_left = self.just_finished_charge_steps_left
        self.preprocessor.phase_safety_margin = self.SAFETY_MARGIN
        feature, legal_action, reward, critic_extra = self.preprocessor.feature_process(env_obs, self.last_action)
        if self.just_finished_charge_steps_left > 0 and not self.charge_mode:
            self.just_finished_charge_steps_left -= 1
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
            critic_feature=list(critic_extra),
        )
        return obs_data, {}

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action."""
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data):
        """Stochastic inference for training."""
        obs_data = list_obs_data[0]
        return [self._select_action_with_charge_override(obs_data)]

    def exploit(self, env_obs):
        """Greedy inference for evaluation."""
        obs_data, _ = self.observation_process(env_obs)

        feature_t = torch.tensor(
            np.array([obs_data.feature], dtype=np.float32), device=self.device
        ).view(1, Config.DIM_OF_OBSERVATION)
        legal_t = torch.tensor(
            np.array([obs_data.legal_action], dtype=np.float32), device=self.device
        ).view(1, Config.ACTION_NUM)

        self.model.set_eval_mode()
        with torch.no_grad():
            probs = self.model.actor_forward(feature_t, legal_t).cpu().numpy()[0]

        rl_action = int(np.argmax(probs))

        need_charge, bfs_action, fallback_center = self._should_force_charge()
        if self.charge_mode and self.preprocessor.battery == self.preprocessor.battery_max:
            self.charge_mode = False
            self.just_finished_charge_steps_left = self.JUST_FINISHED_CHARGE_STEPS
        if self.charge_mode and self._can_finish_without_charge():
            self.charge_mode = False
        if not self.charge_mode and need_charge:
            self.charge_mode = True
            self.just_finished_charge_steps_left = 0

        action = rl_action
        if self.charge_mode:
            charge_action = self._select_charge_mode_action(obs_data.legal_action, bfs_action, fallback_center)
            if charge_action != -1:
                action = charge_action

        hero_x, hero_z = self.preprocessor.cur_pos
        action = npc_safe_filter(
            action,
            hero_x,
            hero_z,
            self.preprocessor._npcs,
            obs_data.legal_action,
            safe_dist=self._charge_safe_dist() if self.charge_mode else 2,
        )
        action = self._force_charge_progress_if_critical(action, obs_data.legal_action)

        self.last_action = action
        return action

    def _select_action_with_charge_override(self, obs_data):
        """Run PPO policy, apply BFS charge override, then NPC safety filter."""
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        critic_feature = obs_data.critic_feature

        prob, value = self._run_model(feature, legal_action, critic_feature)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        need_charge, bfs_action, fallback_center = self._should_force_charge()

        if self.logger:
            hero_x, hero_z = self.preprocessor.cur_pos
            bfs_dist, _ = bfs_to_charger(
                self.preprocessor,
                hero_x,
                hero_z,
                self.preprocessor._npcs,
                npc_block_radius=self._charge_npc_block_radius(),
            )
            nearest_charger_dist, _, _ = self.preprocessor._nearest_charger((hero_x, hero_z))
            self.logger.info(
                "charge_override battery=%s bfs_dist=%s nearest_charger_dist=%s bfs_action=%s charge_mode=%s",
                self.preprocessor.battery,
                bfs_dist,
                nearest_charger_dist,
                bfs_action,
                self.charge_mode,
            )

        if self.charge_mode and self.preprocessor.battery == self.preprocessor.battery_max:
            self.charge_mode = False
            self.just_finished_charge_steps_left = self.JUST_FINISHED_CHARGE_STEPS
        if self.charge_mode and self._can_finish_without_charge():
            self.charge_mode = False
        if not self.charge_mode and need_charge:
            self.charge_mode = True
            self.just_finished_charge_steps_left = 0

        if self.charge_mode:
            charge_action = self._select_charge_mode_action(legal_action, bfs_action, fallback_center)
            if charge_action != -1:
                action = charge_action
                d_action = charge_action

        hero_x, hero_z = self.preprocessor.cur_pos
        safe_dist = self._charge_safe_dist() if self.charge_mode else 2
        action = npc_safe_filter(action, hero_x, hero_z, self.preprocessor._npcs, legal_action, safe_dist=safe_dist)
        d_action = npc_safe_filter(d_action, hero_x, hero_z, self.preprocessor._npcs, legal_action, safe_dist=safe_dist)
        action = self._force_charge_progress_if_critical(action, legal_action)
        d_action = self._force_charge_progress_if_critical(d_action, legal_action)

        return ActData(
            action=[action],
            d_action=[d_action],
            prob=list(prob),
            value=value,
        )

    def _should_force_charge(self):
        """Decide whether battery level requires forced return to charger."""
        hero_x, hero_z = self.preprocessor.cur_pos
        bfs_dist, bfs_action = bfs_to_charger(
            self.preprocessor,
            hero_x,
            hero_z,
            self.preprocessor._npcs,
            npc_block_radius=self._charge_npc_block_radius(),
        )
        fallback_center, nearest_charger_dist = self._nearest_known_charger_center()

        if self._can_finish_without_charge():
            self.charge_trigger_count = 0
            return False, bfs_action, fallback_center

        battery = self.preprocessor.battery
        raw_trigger = battery <= bfs_dist + self.SAFETY_MARGIN
        critical_trigger = battery <= bfs_dist + self.CHARGE_CRITICAL_BUFFER
        fallback_trigger = (
            fallback_center is not None
            and battery <= nearest_charger_dist * self.DIRECT_CHARGE_CHEB_MULTIPLIER + self.DIRECT_CHARGE_MARGIN
        )

        if self.charge_mode:
            keep_charge = battery <= bfs_dist + self.SAFETY_MARGIN + self.CHARGE_RELEASE_BUFFER or fallback_trigger
            return keep_charge, bfs_action, fallback_center

        if raw_trigger:
            self.charge_trigger_count += 1
        else:
            self.charge_trigger_count = 0

        need_charge = critical_trigger or fallback_trigger or self.charge_trigger_count >= self.CHARGE_TRIGGER_CONFIRM_STEPS
        return need_charge, bfs_action, fallback_center

    def _can_finish_without_charge(self):
        remaining_steps = max(0, self.EPISODE_MAX_STEP - int(self.preprocessor.step_no))
        return self.preprocessor.battery >= remaining_steps

    def _nearest_known_charger_center(self):
        hero_x, hero_z = self.preprocessor.cur_pos
        if not self.preprocessor._chargers:
            return None, float("inf")

        best_center = None
        best_dist = float("inf")
        for charger in self.preprocessor._chargers:
            center = self.preprocessor._charger_center(charger)
            dist = max(abs(center[0] - hero_x), abs(center[1] - hero_z))
            if dist < best_dist:
                best_dist = dist
                best_center = center
        return best_center, best_dist

    def _is_low_battery_charge_state(self):
        return self.preprocessor.battery / max(float(self.preprocessor.battery_max), 1.0) <= self.CHARGE_LOW_BATTERY_RATIO

    def _charge_npc_block_radius(self):
        return self.CHARGE_URGENT_NPC_BLOCK_RADIUS if self._is_low_battery_charge_state() else 2

    def _charge_safe_dist(self):
        return self.CHARGE_URGENT_SAFE_DIST if self._is_low_battery_charge_state() else 2

    def _select_charge_mode_action(self, legal_action, bfs_action, fallback_center):
        if 0 <= bfs_action < len(legal_action) and legal_action[bfs_action]:
            return bfs_action
        return self._select_pessimistic_charge_action(legal_action, fallback_center)

    def _select_pessimistic_charge_action(self, legal_action, target_center):
        if target_center is None:
            return -1

        hero_x, hero_z = self.preprocessor.cur_pos
        target_x, target_z = target_center
        current_dist = max(abs(target_x - hero_x), abs(target_z - hero_z))
        best_action = -1
        best_score = float("-inf")

        for action_idx in range(len(legal_action)):
            if not legal_action[action_idx]:
                continue

            dx, dz = self._action_delta(action_idx)
            next_x = hero_x + dx
            next_z = hero_z + dz
            if not self.preprocessor.is_known_passable(next_x, next_z):
                continue

            next_dist = max(abs(target_x - next_x), abs(target_z - next_z))
            if next_dist > current_dist:
                continue

            score = -4.0 * next_dist
            if next_dist < current_dist:
                score += 6.0
            if self._nearest_npc_dist(next_x, next_z) < self._charge_safe_dist():
                score -= 8.0
            if dx == 0 or dz == 0:
                score += 0.5
            if score > best_score:
                best_score = score
                best_action = action_idx

        return best_action

    def _force_charge_progress_if_critical(self, action, legal_action):
        if not self.charge_mode:
            return action

        hero_x, hero_z = self.preprocessor.cur_pos
        current_dist, _ = bfs_to_charger(
            self.preprocessor,
            hero_x,
            hero_z,
            self.preprocessor._npcs,
            npc_block_radius=self._charge_npc_block_radius(),
        )
        if current_dist == float("inf"):
            return action

        if self.preprocessor.battery > current_dist + self.CHARGE_CRITICAL_BUFFER:
            return action

        if self._action_reduces_charge_dist(action, legal_action, current_dist):
            return action

        best_action = -1
        best_dist = current_dist
        for action_idx in range(len(legal_action)):
            if not self._action_reduces_charge_dist(action_idx, legal_action, current_dist):
                continue

            dx, dz = self._action_delta(action_idx)
            next_dist, _ = bfs_to_charger(
                self.preprocessor,
                hero_x + dx,
                hero_z + dz,
                self.preprocessor._npcs,
                npc_block_radius=self._charge_npc_block_radius(),
            )
            if next_dist < best_dist:
                best_dist = next_dist
                best_action = action_idx

        return best_action if best_action != -1 else action

    def _action_reduces_charge_dist(self, action, legal_action, current_dist):
        if not (0 <= action < len(legal_action)) or not legal_action[action]:
            return False

        dx, dz = self._action_delta(action)
        hero_x, hero_z = self.preprocessor.cur_pos
        next_x = hero_x + dx
        next_z = hero_z + dz
        if not self.preprocessor.is_passable(next_x, next_z):
            return False

        next_dist, _ = bfs_to_charger(
            self.preprocessor,
            next_x,
            next_z,
            self.preprocessor._npcs,
            npc_block_radius=self._charge_npc_block_radius(),
        )
        return next_dist < current_dist

    def _nearest_npc_dist(self, x, z):
        if not self.preprocessor._npcs:
            return float("inf")
        best = float("inf")
        for npc in self.preprocessor._npcs:
            pos = npc.get("pos", {})
            npc_x = int(pos.get("x", 0))
            npc_z = int(pos.get("z", 0))
            best = min(best, max(abs(x - npc_x), abs(z - npc_z)))
        return best

    @staticmethod
    def _action_delta(action):
        return (
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        )[action]

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update."""
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint."""
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint."""
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _run_model(self, feature, legal_action, critic_feature):
        """Gradient-free full forward pass (actor + critic)."""
        self.model.set_eval_mode()
        actor_tensor = torch.tensor(
            np.array([feature], dtype=np.float32), device=self.device
        ).view(1, Config.DIM_OF_OBSERVATION)
        legal_tensor = torch.tensor(
            np.array([legal_action], dtype=np.float32), device=self.device
        ).view(1, Config.ACTION_NUM)
        critic_tensor = torch.tensor(
            np.array([critic_feature], dtype=np.float32), device=self.device
        ).view(1, Config.CRITIC_EXTRA_DIM)

        with torch.no_grad():
            rst = self.model(actor_tensor, legal_tensor, critic_tensor, inference=True)

        probs = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return probs, value

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True)."""
        if use_max:
            return int(np.argmax(probs))
        p = probs.astype(np.float64)
        p = p / p.sum()
        return int(np.argmax(np.random.multinomial(1, p, size=1)))
