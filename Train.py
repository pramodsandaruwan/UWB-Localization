import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

WINDOW = 25
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-4

ANCHORS = [100, 101, 102, 103]

ANCHOR_POS = {
    100: np.array([0.0,   0.0,   1.17]),
    101: np.array([10.73, 0.0,   0.54]),
    102: np.array([-0.27, 23.61, 0.52]),
    103: np.array([10.87, 22.72, 0.75]),
}

def build_features(imu, uwb):

    gyro = imu[0:3] / 10.0
    acc  = imu[3:6] / 20.0
    quat = imu[6:10]

    feats = list(gyro) + list(acc) + list(quat)

    feats += [np.linalg.norm(gyro), np.linalg.norm(acc)]

    for m in ANCHORS:

        r = uwb.get(m, {}).get("range", 0.0)

        if r <= 0 or r > 50:
            r = 0.0
        else:
            r = r / 30.0

        feats.append(r)

    return np.array(feats, dtype=np.float32)

def load_data(imu_file, uwb_file, gt_file):

    imu = pd.read_csv(imu_file)
    uwb = pd.read_csv(uwb_file)
    gt  = pd.read_csv(gt_file)

    imu = imu.rename(columns={"timestamp_sec": "time"})

    uwb = uwb.rename(columns={
        "timestamp_sec": "time",
        "module_id": "anchor_id",
        "range_m": "range"
    })

    gt = gt.rename(columns={"timestamp_sec": "time"})

    return (
        imu.sort_values("time"),
        uwb.sort_values("time"),
        gt.sort_values("time")
    )

def nearest_row(df, t):

    idx = (df["time"] - t).abs().idxmin()
    return df.loc[idx]

def get_uwb_snapshot(df, t):

    subset = df.iloc[(df["time"] - t).abs().argsort()[:10]]

    data = {}

    for _, r in subset.iterrows():

        aid = int(r["anchor_id"])
        data[aid] = {"range": r["range"]}

    return data

def trilateration(uwb):

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

def trust_from_error(err):

    if err is None or not np.isfinite(err):
        return 0.0

    err = np.clip(err, 0, 10)

    return np.exp(-(err**2) / (2 * 1.0**2))

def compute_trust(uwb, gt_pos):

    est = trilateration(uwb)

    if est is None:
        return np.zeros(len(ANCHORS)), 0.0

    global_err = np.linalg.norm(est - gt_pos)
    global_trust = trust_from_error(global_err)

    anchor_trust = []

    for m in ANCHORS:

        if m not in uwb:
            anchor_trust.append(0.0)
            continue

        measured = uwb[m]["range"]

        if measured <= 0 or measured > 50:
            anchor_trust.append(0.0)
            continue

        true_d = np.linalg.norm(gt_pos - ANCHOR_POS[m])
        err = abs(measured - true_d)

        anchor_trust.append(
            np.exp(-(err**2) / (2 * 0.5**2))
        )

    return np.array(anchor_trust), global_trust

class FusionDataset(Dataset):

    def __init__(self, imu_file, uwb_file, gt_file):

        imu_df, uwb_df, gt_df = load_data(
            imu_file,
            uwb_file,
            gt_file
        )

        self.samples = []

        buffer = []

        for _, row in imu_df.iterrows():

            t = row["time"]

            imu = row[
                [
                    "ang_vel_x",
                    "ang_vel_y",
                    "ang_vel_z",
                    "lin_acc_x",
                    "lin_acc_y",
                    "lin_acc_z",
                    "quat_x",
                    "quat_y",
                    "quat_z",
                    "quat_w"
                ]
            ].values.astype(float)

            uwb = get_uwb_snapshot(uwb_df, t)

            gt = nearest_row(gt_df, t)

            gt_pos = gt[
                ["gt_x", "gt_y", "gt_z"]
            ].values.astype(float)

            feat = build_features(imu, uwb)

            buffer.append(feat)

            if len(buffer) < WINDOW:
                continue

            x = np.array(
                buffer[-WINDOW:],
                dtype=np.float32
            )

            anchor_trust, global_trust = compute_trust(
                uwb,
                gt_pos
            )

            y = np.concatenate(
                [anchor_trust, [global_trust]]
            ).astype(np.float32)

            self.samples.append((x, y))

            buffer.pop(0)

        print("Total samples:", len(self.samples))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, i):

        x, y = self.samples[i]

        return torch.tensor(x), torch.tensor(y)

class TrustTCN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(
                input_size,
                64,
                kernel_size=3,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                64,
                kernel_size=3,
                padding=4,
                dilation=2
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=8,
                dilation=4
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.uwb = nn.Linear(128, 4)

        self.global_t = nn.Linear(128, 1)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.net(x).squeeze(-1)

        uwb_out = torch.sigmoid(self.uwb(x))

        global_out = torch.sigmoid(self.global_t(x))

        return uwb_out, global_out

def to_class(x):

    if x < 0.3:
        return 0

    elif x < 0.7:
        return 1

    return 2

def evaluate_confusion(model, loader):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for X, Y in loader:

            _, global_pred = model(X)

            pred = global_pred.squeeze().cpu().numpy()

            true = Y[:, 4].cpu().numpy()

            for p, t in zip(pred, true):

                y_pred.append(to_class(p))

                y_true.append(to_class(t))

    return np.array(y_true), np.array(y_pred)

def train():

    dataset = FusionDataset(
        "imu2.csv",
        "uwb2.csv",
        "gt2.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    input_size = dataset.samples[0][0].shape[1]

    model = TrustTCN(input_size)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    loss_fn = nn.MSELoss()

    losses = []

    for ep in range(EPOCHS):

        total = 0

        for X, Y in loader:

            uwb_pred, global_pred = model(X)

            loss = loss_fn(
                global_pred,
                Y[:, 4:5]
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total += loss.item()

        avg = total / len(loader)

        losses.append(avg)

        print(f"Epoch {ep + 1}: {avg:.6f}")

    y_true, y_pred = evaluate_confusion(
        model,
        loader
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    disp.plot(cmap="Blues")

    plt.title("Trust Confusion Matrix")

    plt.show()

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": input_size
        },
        "trust_tcn.pth"
    )

    plt.plot(losses)

    plt.title("Training Loss")

    plt.grid()

    plt.show()

if __name__ == "__main__":

    train()