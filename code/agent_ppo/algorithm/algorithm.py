#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Standard PPO algorithm for Robot Vacuum.
清扫大作战 PPO 算法。

Loss composition / 损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
"""

import os
import time

import torch

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START
        self.label_size = Config.ACTION_NUM

        self.train_step = 0
        self.last_report_time = 0

    def learn(self, list_sample_data):
        """Training entry: perform one PPO gradient step on a batch of SampleData.

        训练入口：接收一批 SampleData，执行一步梯度更新。
        """
        obs = torch.stack([s.obs for s in list_sample_data]).to(self.device)
        critic_obs = torch.stack([s.critic_obs for s in list_sample_data]).to(self.device)
        legal_action = torch.stack([s.legal_action for s in list_sample_data]).to(self.device)
        act = torch.stack([s.act for s in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([s.prob for s in list_sample_data]).to(self.device)
        old_value = torch.stack([s.value for s in list_sample_data]).to(self.device)
        reward_sum = torch.stack([s.reward_sum for s in list_sample_data]).to(self.device)
        advantage = torch.stack([s.advantage for s in list_sample_data]).to(self.device)
        reward = torch.stack([s.reward for s in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(obs, legal_action, critic_obs)
        prob_dist, value_pred = rst_list[0], rst_list[1]

        total_loss, info = self._compute_loss(
            prob_dist=prob_dist,
            value_pred=value_pred,
            old_action=act,
            old_prob=old_prob,
            old_value=old_value,
            reward_sum=reward_sum,
            advantage=advantage,
        )

        total_loss.backward()

        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        results = {"total_loss": total_loss.item()}

        # Periodic monitoring report
        # 定期上报监控
        now = time.time()
        if now - self.last_report_time >= 60:
            results["value_loss"] = round(info["value_loss"], 4)
            results["policy_loss"] = round(info["policy_loss"], 4)
            results["entropy_loss"] = round(info["entropy_loss"], 4)
            results["reward"] = round(reward.mean().item(), 4)

            self.logger.info(
                f"policy_loss: {results['policy_loss']}, "
                f"value_loss: {results['value_loss']}, "
                f"entropy_loss: {results['entropy_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_time = now

        return results

    def _compute_loss(self, prob_dist, value_pred, old_action, old_prob, old_value, reward_sum, advantage):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 三项损失。模型已输出 softmax 概率，无需再做 masked_softmax。
        """
        # Value loss (plain MSE, no clipping)
        # E7.4: removed value clipping — clip_param=0.2 is far too tight for
        # return scale ~30-80, preventing the critic from catching up.
        # 价值损失（普通 MSE，去掉裁剪）
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred

        value_loss = 0.5 * ((tdret - vp) ** 2).mean()

        # Policy loss (PPO clip)
        # 策略损失（PPO clip）
        entropy_loss = (-(prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1)).mean()

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True)

        ratio = new_prob / old_action_prob.clamp(1e-9)

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = adv.unsqueeze(-1)

        policy_loss = torch.maximum(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        # Total loss
        # 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
