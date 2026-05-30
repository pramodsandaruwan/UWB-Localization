#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

class TrustTCN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_size, 64, 3, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=8, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.uwb = nn.Linear(128, 4)
        self.global_t = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.net(x).squeeze(-1)

        return (
            torch.sigmoid(self.uwb(x)),
            torch.sigmoid(self.global_t(x))
        )


# FEATURE FUNCTION
def build_features(imu, uwb, anchors=[100,101,102,103]):

    gyro = imu[0:3] / 10.0
    acc  = imu[3:6] / 20.0
    quat = imu[6:10]

    feats = list(gyro) + list(acc) + list(quat)
    feats += [np.linalg.norm(gyro), np.linalg.norm(acc)]

    for m in anchors:
        r = uwb.get(m, {}).get("range", 0.0)
        if r <= 0 or r > 50:
            r = 0.0
        else:
            r = r / 30.0
        feats.append(r)

    return np.array(feats, dtype=np.float32)