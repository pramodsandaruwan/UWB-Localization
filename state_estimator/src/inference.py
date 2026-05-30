#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import TrustTCN, build_features

ANCHORS = [100, 101, 102, 103]

ANCHOR_POS = {
    100: np.array([0.0,   0.0,   0.99]),
    101: np.array([12.09, 0.0,   1.60]),
    102: np.array([-0.66, 12.99, 1.41]),
    103: np.array([11.62, 13.06, 1.19]),
}

WINDOW = 25


class UKF:

    def __init__(self, dt=0.1):
        self.dt = dt
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
        self.R = np.eye(3) * 0.8

    def set_measurement_noise(self, weights):
        trust = np.max(weights)
        uncertainty = 1.0 - trust
        base = 0.3
        scale = 2.0
        r_val = base + scale * uncertainty
        self.R = np.eye(3) * r_val

    def predict(self, acc):
        dt = self.dt

        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        B = np.zeros((6, 3))
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        self.x[3:] *= 0.98
        self.x = F @ self.x + B @ acc
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_pos(self):
        return self.x[:3]


def raw_trilat(uwb):
    pts = []
    dists = []

    for m in ANCHORS:
        if m not in uwb:
            continue
        r = uwb[m]["range"]
        if r <= 0 or r > 50:
            continue
        pts.append(ANCHOR_POS[m])
        dists.append(r)

    if len(pts) < 3:
        return None

    pts = np.array(pts)
    dists = np.array(dists)

    A = []
    b = []
    p1 = pts[0]
    d1 = dists[0]

    for i in range(1, len(pts)):
        pi = pts[i]
        di = dists[i]
        A.append(2 * (pi - p1))
        b.append(d1**2 - di**2 - np.sum(p1**2) + np.sum(pi**2))

    A = np.array(A)
    b = np.array(b)

    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        return sol
    except:
        return None


def weighted_trilat(uwb, weights):
    pts = []
    dists = []
    wts = []

    weights = np.array(weights)
    weights = np.clip(weights, 0.05, None)
    weights = weights / (np.sum(weights) + 1e-8)

    for i, m in enumerate(ANCHORS):
        if m not in uwb:
            continue
        pts.append(ANCHOR_POS[m])
        dists.append(uwb[m]["range"])
        wts.append(weights[i])

    if len(pts) < 3:
        return None

    pts = np.array(pts)
    dists = np.array(dists)
    wts = np.array(wts)

    A = []
    b = []
    p1 = pts[0]
    d1 = dists[0]

    for i in range(1, len(pts)):
        pi = pts[i]
        di = dists[i]
        wi = wts[i]
        A.append(wi * 2 * (pi - p1))
        b.append(wi * (d1**2 - di**2 - np.sum(p1**2) + np.sum(pi**2)))

    A = np.array(A)
    b = np.array(b)

    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        return sol
    except:
        return None


def nearest(df, t):
    valid = df[df["time"].notna()].copy()
    if len(valid) == 0:
        return None
    diff = (valid["time"] - t).abs().dropna()
    if len(diff) == 0:
        return None
    idx = diff.idxmin()
    return valid.loc[idx]


def main():
    imu = pd.read_csv("imu.csv")
    uwb = pd.read_csv("uwb.csv")
    gt = pd.read_csv("gt.csv")

    imu = imu.dropna(subset=["timestamp_sec"])
    uwb = uwb.dropna(subset=["timestamp_sec"])
    gt = gt.dropna(subset=["timestamp_sec"])

    imu["time"] = imu["timestamp_sec"]
    uwb["time"] = uwb["timestamp_sec"]
    gt["time"] = gt["timestamp_sec"]

    checkpoint = torch.load("trust_tcn.pth", map_location="cpu")
    model = TrustTCN(checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    buffer = []
    gt_list = []
    raw_list = []
    tcn_list = []
    ukf_list = []
    trust_history = {a: [] for a in ANCHORS}
    time_list = []

    ukf = UKF(dt=0.1)

    for _, row in imu.iterrows():
        t = row["time"]

        gt_row = nearest(gt, t)
        if gt_row is None:
            continue

        gt_pos = np.array([gt_row["gt_x"], gt_row["gt_y"], gt_row["gt_z"]])

        uwb_snap = {}
        subset = uwb[abs(uwb["time"] - t) < 0.2]
        for _, r in subset.iterrows():
            uwb_snap[int(r["module_id"])] = {"range": float(r["range_m"])}

        imu_vec = row[[
            "ang_vel_x", "ang_vel_y", "ang_vel_z",
            "lin_acc_x", "lin_acc_y", "lin_acc_z",
            "quat_x", "quat_y", "quat_z", "quat_w"
        ]].values.astype(float)

        acc = np.array([row["lin_acc_x"], row["lin_acc_y"], row["lin_acc_z"] - 9.81])

        feat = build_features(imu_vec, uwb_snap)
        buffer.append(feat)
        ukf.predict(acc)

        if len(buffer) < WINDOW:
            continue

        x = torch.tensor(
            np.array(buffer[-WINDOW:], dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            uwb_t, _ = model(x)
            weights = uwb_t.squeeze(0).cpu().numpy()

        for i, a in enumerate(ANCHORS):
            trust_history[a].append(weights[i])

        time_list.append(t)
        ukf.set_measurement_noise(weights)

        tcn_est = weighted_trilat(uwb_snap, weights)
        if tcn_est is not None:
            ukf.update(tcn_est)
        ukf_list.append(ukf.get_pos())

        raw_est = raw_trilat(uwb_snap)
        if raw_est is not None:
            raw_list.append(raw_est)

        if tcn_est is not None:
            tcn_list.append(tcn_est)

        gt_list.append(gt_pos)
        buffer.pop(0)

    raw = np.array(raw_list)
    tcn = np.array(tcn_list)
    gt = np.array(gt_list)
    ukf = np.array(ukf_list)

    n = min(len(raw), len(tcn), len(gt), len(ukf))
    raw, tcn, gt, ukf = raw[:n], tcn[:n], gt[:n], ukf[:n]

    plt.figure()
    plt.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=1)
    plt.plot(raw[:, 0], raw[:, 1], label="Raw UWB", color="tab:orange", alpha=0.6)
    plt.plot(tcn[:, 0], tcn[:, 1], label="TCN Weighted", color="tab:green", alpha=0.7)
    plt.plot(ukf[:, 0], ukf[:, 1], label="UKF Fusion", color="tab:red", linewidth=1)
    plt.title("Final Localization System")
    plt.legend()
    plt.grid()
    plt.show()

    raw_err = np.linalg.norm(raw - gt, axis=1)
    tcn_err = np.linalg.norm(tcn - gt, axis=1)
    ukf_err = np.linalg.norm(ukf - gt, axis=1)

    plt.figure()
    plt.plot(raw_err, label="Raw UWB", color="tab:orange")
    plt.plot(tcn_err, label="TCN Weighted", color="tab:green")
    plt.plot(ukf_err, label="UKF Fusion", color="tab:red")
    plt.title("Error Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    print("RAW:", np.mean(raw_err))
    print("TCN:", np.mean(tcn_err))
    print("UKF:", np.mean(ukf_err))

    plt.figure()
    for a in ANCHORS:
        plt.plot(time_list[:len(trust_history[a])], trust_history[a], label=f"A{a}")
    plt.title("TCN Anchor Trust Evolution")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()