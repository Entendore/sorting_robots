"""sorting_dqn.py — DQN e-puck sorter with hue-based colour perception.

Continuous 10-D features (position, heading, bump, hue-circle colour
encoding, blob offset & size) into a small numpy MLP trained with replay
+ target network. Hue encoding lets the trained policy route *unseen*
colours (e.g. orange) to the nearest learned zone — the generalization
experiment. Logs episodes to learning_log_<robot>.csv (dashboard-ready).
"""

import csv
import math
import os
import random

import numpy as np
from controller import Supervisor

RED, BLUE, NO = 1, 2, 0
ZONES = {RED: (0.75, 0.0, 0.18), BLUE: (-0.75, 0.0, 0.18)}
BALL_DEFS = {RED: ["RED_BALL_1", "RED_BALL_2", "RED_BALL_3"],
             BLUE: ["BLUE_BALL_1", "BLUE_BALL_2", "BLUE_BALL_3"]}

N_IN, N_HID, N_OUT = 10, 32, 3
LR, GAMMA = 1e-3, 0.9
BUF_SIZE, BATCH, WARMUP, TARGET_SYNC = 20000, 32, 400, 300
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.97
ACTION_STEPS, MAX_STEPS, EPISODES, EVAL_N = 20, 200, 200, 8
FWD_SPEED, TURN_SPEED = 3.0, 2.0
BUMP_THRESHOLD = 600


class MLP:
    """N_IN -> ReLU(N_HID) -> N_OUT Q-values, trained on Bellman targets."""

    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0.0, np.sqrt(2.0 / N_IN), (N_IN, N_HID))
        self.b1 = np.zeros(N_HID)
        self.W2 = r.normal(0.0, np.sqrt(2.0 / N_HID), (N_HID, N_OUT))
        self.b2 = np.zeros(N_OUT)

    def q(self, X):
        self.X = X
        self.H = np.maximum(X @ self.W1 + self.b1, 0.0)
        return self.H @ self.W2 + self.b2

    def copy_from(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()

    def fit(self, X, Y):
        err = np.clip(self.q(X) - Y, -1.0, 1.0)
        gW2 = self.H.T @ err / len(X)
        gb2 = err.mean(0)
        dH = err @ self.W2.T
        dH[self.H <= 0] = 0.0
        gW1 = self.X.T @ dH / len(X)
        gb1 = dH.mean(0)
        self.W1 -= LR * gW1; self.b1 -= LR * gb1
        self.W2 -= LR * gW2; self.b2 -= LR * gb2


class DqnSorter(Supervisor):

    def __init__(self):
        super(DqnSorter, self).__init__()
        self.ts = int(self.getBasicTimeStep())
        self.name_ = self.getName()
        self.left = self.getDevice("left wheel motor")
        self.right = self.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
        self.sensors = [self.getDevice("ps%d" % i) for i in range(8)]
        for s in self.sensors:
            s.enable(2 * self.ts)
        self.camera = self.getDevice("camera")
        self.camera.enable(2 * self.ts)

        self.balls = {}
        for colour, names in BALL_DEFS.items():
            for n in names:
                node = self.getFromDef(n)
                self.balls[n] = {"field": node.getField("translation"),
                                 "spawn": node.getField("translation")
                                 .getSFVec3f(),
                                 "colour": colour, "done": False}

        self.online, self.target = MLP(), MLP(seed=1)
        self.buffer, self.updates = [], 0
        self.eps, self.episode = EPS_START, 0
        self.ep_ok, self.ep_bad = 0, 0
        if not self.load():
            print("[%s] fresh network" % self.name_)

    # ------------------------------------------------------ perception ---
    def blob(self):
        """Dominant colour as (hue in [-1,1], brightness) + offset/size."""
        w, h = self.camera.getWidth(), self.camera.getHeight()
        img = self.camera.getImageArray()
        if not img:
            return 0.0, 0.0, 0.0, 0.0
        cnt, sx, sr, sg, sb = 0, 0.0, 0.0, 0.0, 0.0
        for y in range(h // 4, h, 2):
            row = img[y]
            for x in range(0, w, 2):
                r, g, b = row[x][0], row[x][1], row[x][2]
                if max(r, g, b) > 90 and max(r, g, b) - min(r, g, b) > 35:
                    cnt += 1
                    sx += x
                    sr += r; sg += g; sb += b
        if cnt == 0:
            return 0.0, 0.0, 0.0, 0.0
        r, g, b = sr / cnt, sg / cnt, sb / cnt
        hue = math.atan2(1.732 * (g - b), 2 * r - g - b) / math.pi
        bright = (r + g + b) / (3.0 * 255.0)
        return hue, bright, (sx / cnt / (w - 1)) * 2 - 1, cnt * 4.0 / (w * h)

    def features(self):
        p = self.getSelf().getField("translation").getSFVec3f()
        rot = self.getSelf().getField("rotation").getSFRotation()
        yaw = rot[3] * (1.0 if rot[1] >= 0 else -1.0)
        bump = 1.0 if max(s.getValue() for s in self.sensors) > BUMP_THRESHOLD \
            else 0.0
        hue, bright, dx, frac = self.blob()
        return np.array([p[0], p[2], math.sin(yaw), math.cos(yaw), bump,
                         math.sin(hue * math.pi), math.cos(hue * math.pi),
                         bright, dx, frac], dtype=np.float32)

    # ----------------------------------------------------------- env -----
    def set_action(self, a):
        lv, rv = ((FWD_SPEED, FWD_SPEED) if a == 0 else
                  (-TURN_SPEED, TURN_SPEED) if a == 1 else
                  (TURN_SPEED, -TURN_SPEED))
        self.left.setVelocity(lv)
        self.right.setVelocity(rv)

    def zone_events(self):
        r = 0.0
        for info in self.balls.values():
            if info["done"]:
                continue
            x, _, z = info["field"].getSFVec3f()
            for colour, (zx, zz, half) in ZONES.items():
                if abs(x - zx) <= half and abs(z - zz) <= half:
                    info["done"] = True
                    if colour == info["colour"]:
                        r += 10.0
                        self.ep_ok += 1
                    else:
                        r -= 10.0
                        self.ep_bad += 1
        return r, all(i["done"] for i in self.balls.values())

    def reset_episode(self):
        for info in self.balls.values():
            sx, y, sz = info["spawn"]
            info["field"].setSFVec3f([sx + random.uniform(-0.03, 0.03), y,
                                      sz + random.uniform(-0.03, 0.03)])
            info["done"] = False
        self.ep_ok, self.ep_bad = 0, 0

    # ---------------------------------------------------------- learn ----
    def learn_step(self):
        batch = random.sample(self.buffer, BATCH)
        X = np.array([t[0] for t in batch])
        X2 = np.array([t[3] for t in batch])
        max_q2 = self.target.q(X2).max(axis=1)
        Y = self.online.q(X).copy()
        for i, (_, a, r, _, done) in enumerate(batch):
            Y[i, a] = r + (0.0 if done else GAMMA * max_q2[i])
        self.online.fit(X, Y)
        self.updates += 1
        if self.updates % TARGET_SYNC == 0:
            self.target.copy_from(self.online)

    def run_episode(self, greedy=False):
        self.reset_episode()
        total = 0.0
        for _ in range(MAX_STEPS):
            s = self.features()
            if not greedy and random.random() < self.eps:
                a = random.randrange(N_OUT)
            else:
                a = int(np.argmax(self.online.q(s[None, :])[0]))
            self.set_action(a)
            for _ in range(ACTION_STEPS):
                if self.step(2 * self.ts) == -1:
                    return None
            zr, done = self.zone_events()
            s2 = self.features()
            r = -0.02 + zr + (0.2 if s2[9] > 0.01 else 0.0)
            if not greedy:
                self.buffer.append((s, a, r, s2, done))
                if len(self.buffer) > BUF_SIZE:
                    self.buffer.pop(0)
                if len(self.buffer) >= WARMUP:
                    self.learn_step()
            total += r
            if done:
                break
        self.left.setVelocity(0.0)
        self.right.setVelocity(0.0)
        return total

    # ------------------------------------------------------------- io ----
    def folder(self):
        return os.path.dirname(os.path.abspath(__file__))

    def weights_path(self):
        return os.path.join(self.folder(),
                            "dqn_%s.npz" % self.name_.replace(" ", "_"))

    def save(self):
        np.savez(self.weights_path(), W1=self.online.W1, b1=self.online.b1,
                 W2=self.online.W2, b2=self.online.b2)

    def load(self):
        try:
            d = np.load(self.weights_path())
            self.online.W1, self.online.b1 = d["W1"], d["b1"]
            self.online.W2, self.online.b2 = d["W2"], d["b2"]
            self.target.copy_from(self.online)
            return True
        except (OSError, KeyError):
            return False

    def log_episode(self, total):
        path = os.path.join(self.folder(),
                            "learning_log_%s.csv" % self.name_.replace(" ", "_"))
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["robot", "episode", "return",
                                "correct", "wrong", "epsilon"])
                w.writerow([self.name_, self.episode, "%.2f" % total,
                            self.ep_ok, self.ep_bad, "%.3f" % self.eps])
        except OSError as e:
            print("[%s] log write failed: %s" % (self.name_, e))

    # ------------------------------------------------------------ main ---
    def run(self):
        print("[%s] DQN training started (hue-encoded perception)"
              % self.name_)
        while True:
            result = self.run_episode()
            if result is None:
                self.save()
                break
            self.episode += 1
            self.log_episode(result)
            if self.episode % 10 == 0:
                self.save()
            self.eps = max(EPS_END, self.eps * EPS_DECAY)
            if self.episode % 5 == 0:
                print("[%s] episode %3d | return %7.2f | sorted %d/%d "
                      "| eps %.2f"
                      % (self.name_, self.episode, result, self.ep_ok,
                         self.ep_ok + self.ep_bad, self.eps))
            if self.episode == EPISODES:
                self.save()
                scores = [self.run_episode(greedy=True) for _ in range(EVAL_N)]
                scores = [s for s in scores if s is not None]
                print("[%s] greedy eval: mean return %.1f over %d episodes"
                      % (self.name_, sum(scores) / max(1, len(scores)),
                         len(scores)))
                break


if __name__ == "__main__":
    DqnSorter().run()
