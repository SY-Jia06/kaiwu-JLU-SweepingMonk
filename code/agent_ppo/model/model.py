#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Separated actor/critic MLP for Robot Vacuum.
清扫大作战分离 Actor/Critic MLP 策略网络。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """Actor/Critic MLP with no shared backbone."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        actor_dim = Config.DIM_OF_OBSERVATION
        critic_dim = Config.CRITIC_INPUT_DIM
        act_num = Config.ACTION_NUM

        self.actor = nn.Sequential(
            _make_fc(actor_dim, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
            _make_fc(64, act_num, gain=0.01),
        )

        self.critic = nn.Sequential(
            _make_fc(critic_dim, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
            _make_fc(64, 1, gain=1.0),
        )

    def forward(self, actor_input, legal_action, critic_extra=None, inference=False):
        """Forward actor and critic with separate inputs."""
        actor_x = actor_input.to(torch.float32)
        legal = legal_action.to(torch.float32)
        logits = self.actor(actor_x)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(legal == 0, -1e8)
        probs = torch.softmax(logits, dim=-1)

        if critic_extra is None:
            critic_extra = torch.zeros(
                actor_x.size(0),
                Config.CRITIC_EXTRA_DIM,
                device=actor_x.device,
                dtype=torch.float32,
            )
        critic_input = torch.cat([actor_x, critic_extra.to(torch.float32)], dim=-1)
        value = self.critic(critic_input)
        return [probs, value]

    def actor_forward(self, actor_input, legal_action):
        """Actor-only forward for evaluation (no critic computation)."""
        actor_x = actor_input.to(torch.float32)
        legal = legal_action.to(torch.float32)
        logits = self.actor(actor_x)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(legal == 0, -1e8)
        return torch.softmax(logits, dim=-1)

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
