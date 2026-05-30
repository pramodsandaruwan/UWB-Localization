#!/usr/bin/env python3

import threading
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String as StringMsg

from model import TrustTCN, build_features

# Configuration

ANCHORS = [100, 101, 102, 103]

ANCHOR_POS = {
    100: np.array([0.0,   0.0,   0.99]),
    101: np.array([12.09, 0.0,   1.60]),
    102: np.array([-0.66, 12.99, 1.41]),
    103: np.array([11.62, 13.06, 1.19]),
}

WINDOW      = 25
UWB_SYNC_DT = 0.2  

# UKF 

class UKF:

    def __init__(self, dt=0.1):
        self.dt = dt
        self.x  = np.zeros(6)
        self.P  = np.eye(6)
        self.Q  = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
        self.R  = np.eye(3) * 0.8

    def set_measurement_noise(self, weights):
        trust       = np.max(weights)
        uncertainty = 1.0 - trust
        self.R      = np.eye(3) * (0.3 + 2.0 * uncertainty)

    def predict(self, acc):
        dt = self.dt
        F  = np.eye(6); F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        B  = np.zeros((6, 3)); B[3, 0] = dt; B[4, 1] = dt; B[5, 2] = dt
        self.x[3:] *= 0.98
        self.x = F @ self.x + B @ acc
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((3, 6)); H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_pos(self):
        return self.x[:3].copy()


# Trilateration helpers 

def raw_trilat(uwb):
    pts, dists = [], []
    for m in ANCHORS:
        if m not in uwb:
            continue
        r = uwb[m]["range"]
        if r <= 0 or r > 50:
            continue
        pts.append(ANCHOR_POS[m]); dists.append(r)
    if len(pts) < 3:
        return None
    pts   = np.array(pts); dists = np.array(dists)
    A, b  = [], []
    p1, d1 = pts[0], dists[0]
    for i in range(1, len(pts)):
        A.append(2 * (pts[i] - p1))
        b.append(d1**2 - dists[i]**2 - np.dot(p1, p1) + np.dot(pts[i], pts[i]))
    try:
        return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    except Exception:
        return None


def weighted_trilat(uwb, weights):
    pts, dists, wts = [], [], []
    weights = np.clip(np.array(weights), 0.05, None)
    weights = weights / (weights.sum() + 1e-8)
    for i, m in enumerate(ANCHORS):
        if m not in uwb:
            continue
        pts.append(ANCHOR_POS[m]); dists.append(uwb[m]["range"]); wts.append(weights[i])
    if len(pts) < 3:
        return None
    pts  = np.array(pts); dists = np.array(dists); wts = np.array(wts)
    A, b = [], []
    p1, d1 = pts[0], dists[0]
    for i in range(1, len(pts)):
        wi = wts[i]
        A.append(wi * 2 * (pts[i] - p1))
        b.append(wi * (d1**2 - dists[i]**2 - np.dot(p1, p1) + np.dot(pts[i], pts[i])))
    try:
        return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    except Exception:
        return None

# Main node

class InferenceNode:

    def __init__(self):
        rospy.init_node("trust_tcn_inference")

        # load model 
        checkpoint  = torch.load("trust_tcn.pth", map_location="cpu")
        self.model  = TrustTCN(checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.ukf    = UKF(dt=0.1)
        self.buffer = [] 

        # Thread-safe UWB store  {module_id: {"range": float, "time": float}}
        self._uwb_lock = threading.Lock()
        self._uwb      = {}

        # Thread-safe GT store   {"pos": np.array, "time": float}
        self._gt_lock  = threading.Lock()
        self._gt       = None

        self.gt_list    = []
        self.raw_list   = []
        self.tcn_list   = []
        self.ukf_list   = []
        self.time_list  = []
        self.trust_hist = {a: [] for a in ANCHORS}

        rospy.Subscriber("/mavros/imu/data",            Imu,         self._imu_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._gt_cb)
        rospy.Subscriber("/uwb/raw",                    StringMsg,   self._uwb_cb)

        rospy.loginfo("InferenceNode ready")

    # Subscribers

    def _gt_cb(self, msg):
        t   = msg.header.stamp.to_sec()
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        with self._gt_lock:
            self._gt = {"pos": pos, "time": t}

    def _uwb_cb(self, msg):
        try:
            parts = msg.data.strip().split(",")
            if len(parts) != 3:
                rospy.logwarn_throttle(5, f"Unexpected UWB message: '{msg.data}'")
                return
            mid = int(parts[0])
            rng = float(parts[1])
            t_now = rospy.Time.now().to_sec()
            with self._uwb_lock:
                self._uwb[mid] = {"range": rng, "time": t_now}
        except ValueError as e:
            rospy.logwarn_throttle(5, f"UWB parse error: {e} — '{msg.data}'")

    def _imu_cb(self, msg):
        t = msg.header.stamp.to_sec()

        # ground truth snapshot 
        with self._gt_lock:
            gt_snap = self._gt
        if gt_snap is None:
            return 

        gt_pos = gt_snap["pos"]

        # UWB snapshot 
        with self._uwb_lock:
            uwb_snap = {
                mid: {"range": d["range"]}
                for mid, d in self._uwb.items()
                if abs(d["time"] - t) < UWB_SYNC_DT
            }

        # IMU vector
        imu_vec = np.array([
            msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.orientation.x,         msg.orientation.y,
            msg.orientation.z,         msg.orientation.w,
        ], dtype=float)

        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z - 9.81,
        ])

        # build feature vector & slide window 
        feat = build_features(imu_vec, uwb_snap)
        self.buffer.append(feat)
        self.ukf.predict(acc)

        if len(self.buffer) < WINDOW:
            return

        x = torch.tensor(
            np.array(self.buffer[-WINDOW:], dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            uwb_t, _ = self.model(x)
            weights   = uwb_t.squeeze(0).cpu().numpy()

        # record trust history 
        for i, a in enumerate(ANCHORS):
            self.trust_hist[a].append(weights[i])

        # localization 
        self.ukf.set_measurement_noise(weights)

        tcn_est = weighted_trilat(uwb_snap, weights)
        if tcn_est is not None:
            self.ukf.update(tcn_est)

        raw_est = raw_trilat(uwb_snap)

        # store results
        self.time_list.append(t)
        self.gt_list.append(gt_pos)
        self.ukf_list.append(self.ukf.get_pos())

        if raw_est is not None:
            self.raw_list.append(raw_est)

        if tcn_est is not None:
            self.tcn_list.append(tcn_est)

        # Trim oldest buffer entry
        self.buffer.pop(0)

    # Plotting — called once on shutdown

    def plot_results(self):
        rospy.loginfo("Generating plots …")

        raw = np.array(self.raw_list)
        tcn = np.array(self.tcn_list)
        gt  = np.array(self.gt_list)
        ukf = np.array(self.ukf_list)

        if any(len(a) == 0 for a in (raw, tcn, gt, ukf)):
            rospy.logwarn("Not enough data to plot — did the node run long enough?")
            return

        n = min(len(raw), len(tcn), len(gt), len(ukf))
        raw, tcn, gt, ukf = raw[:n], tcn[:n], gt[:n], ukf[:n]

        # trajectory 
        fig, ax = plt.subplots()
        ax.plot(gt[:, 0],  gt[:, 1],  label="GT",          linewidth=1.5)
        ax.plot(raw[:, 0], raw[:, 1], label="Raw UWB",      color="tab:orange", alpha=0.6)
        ax.plot(tcn[:, 0], tcn[:, 1], label="TCN Weighted", color="tab:green",  alpha=0.7)
        ax.plot(ukf[:, 0], ukf[:, 1], label="UKF Fusion",   color="tab:red",    linewidth=1.5)
        ax.set_title("Final Localization System"); ax.legend(); ax.grid()
        fig.savefig("trajectory.png", dpi=150)
        rospy.loginfo("Saved trajectory.png")

        # error 
        raw_err = np.linalg.norm(raw - gt, axis=1)
        tcn_err = np.linalg.norm(tcn - gt, axis=1)
        ukf_err = np.linalg.norm(ukf - gt, axis=1)

        fig2, ax2 = plt.subplots()
        ax2.plot(raw_err, label="Raw UWB",      color="tab:orange")
        ax2.plot(tcn_err, label="TCN Weighted",  color="tab:green")
        ax2.plot(ukf_err, label="UKF Fusion",    color="tab:red")
        ax2.set_title("Error Comparison"); ax2.legend(); ax2.grid()
        fig2.savefig("error.png", dpi=150)
        rospy.loginfo("Saved error.png")

        print(f"\n=== Mean positioning error ===")
        print(f"  Raw UWB      : {np.mean(raw_err):.4f} m")
        print(f"  TCN Weighted : {np.mean(tcn_err):.4f} m")
        print(f"  UKF Fusion   : {np.mean(ukf_err):.4f} m")

        # anchor trust 
        t_axis = self.time_list[: min(len(self.time_list),
                                      max(len(v) for v in self.trust_hist.values()))]
        fig3, ax3 = plt.subplots()
        for a in ANCHORS:
            h = self.trust_hist[a]
            ax3.plot(t_axis[:len(h)], h, label=f"A{a}")
        ax3.set_title("TCN Anchor Trust Evolution"); ax3.legend(); ax3.grid()
        fig3.savefig("trust.png", dpi=150)
        rospy.loginfo("Saved trust.png")

        try:
            plt.show()
        except Exception:
            pass


# Entry point

if __name__ == "__main__":

    node = InferenceNode()

    try:
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    finally:
        node.plot_results()