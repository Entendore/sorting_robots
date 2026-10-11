"""ur5e_sorter.py — UR5e cell, v6. FIVE learners, one reward stream:

  sorter Q[4][3]   colour x height-class      -> {red, blue, scrap}  per pick
  speed  Q[6][2]   alarm x cur speed          -> {FAST, SLOW}        per batch
  trust  Q[3][3]   disagreement bucket        -> {A, B, AVG}         per batch
  maint  Q[18][4]  fused-health x spare-health x speed
                                              -> {RUN, CAL_A, CAL_B, CAL_BOTH}
                                                                         Q(lambda)
  opp    Q[3][2]   sicker-gauge bucket        -> {SEIZE(sicker), SKIP}
                                                                         Q(lambda)

Per-gauge economics: CAL_A/CAL_B cost -0.8 (6 s), CAL_BOTH -1.5 (9 s,
batching discount). Fixing a gauge you are not using is waste — it must
learn that too.

Q(lambda): maint/opp decisions get accumulating eligibility traces across
segments; a calibration's payoff (clean measurement in LATER segments)
credits back to the decision that caused it.

Opportunistic maintenance: contamination holds (14 s) can be SEIZEd for a
cheap calibration of the sicker gauge (-0.2 instead of -1.5).

Trust: gauge disagreement (dAB EMA) bucket -> which reading the whole
line fuses. The sign of disagreement does NOT reveal which gauge lies;
the bandit resolves it from grading outcomes.

Policy card: Q-tables + named argmax policy + config fingerprint are
written to policy_card.json every SAVE_EVERY_BATCHES batches and on exit;
a fresh run warm-starts when version and config match (WARM_START flag).
"""

import json
import math
import os
import random

import numpy as np
from controller import Supervisor

# ------------------------------------------------- world constants --------
BASE_XYZ = np.array([0.45, 0.0, 0.0])
BIN = {0: (0.50, 0.42), 1: (0.50, -0.42)}
SCRAP = (0.75, 0.0)
BLOCK_DEFS = {
    "BLOCK_R1": ("red", 0.035), "BLOCK_B1": ("blue", 0.035),
    "BLOCK_R2": ("red", 0.035), "BLOCK_B2": ("blue", 0.035),
    "BLOCK_RB": ("red", 0.025), "BLOCK_BB": ("blue", 0.025),
}
PICK_X, START_X = -0.05, -1.0
QUEUE_GAP = 0.12
PRE_GRASP, PICK_GAP = 0.10, 0.005
TRAVEL_Y, RELEASE_Y = 0.34, 0.15
PARK = (-0.20, 0.32, 0.30)
SCORE_R, SCORE_Y = 0.085, 0.075
BAD_HALF = 0.03
PROV_THRESHOLD = 0.28
STOCK = [(1.06 + 0.14 * (i % 3), 0.14 * (i // 3) - 0.07) for i in range(6)]

# ------------------------------------------------- speeds & economics -----
FAST_V, SLOW_V = 0.06, 0.02
THROUGHPUT_BONUS = 0.8
HOLD_T = 14.0
CONTAM_CHARGE = -3.0
BATCH_PICKS = 6
CAL_G_COST, CAL_BOTH_COST = -0.8, -1.5
T_CAL_ONE, T_CAL_BOTH = 6.0, 9.0     # must match the mast
OPP_COST = -0.2
R_MISS = -0.5
ALPHA, EPS0, EPS_DECAY, EPS_MIN = 0.5, 0.4, 0.96, 0.02

# ------------------------------------------------- lambda machinery -------
GAMMA_META, LAMBDA = 0.95, 0.75
MAINT_ALPHA = OPP_ALPHA = 0.35
TRUST_ALPHA = 0.4
META_ALPHA, META_EPS0 = 0.5, 0.6
META_DECAY, META_EPS_MIN = 0.95, 0.05

# ------------------------------------------------- diagnosis thresholds ---
DRIFT_OK, DRIFT_HIGH = 0.004, 0.010
AGREE_THR = 0.004

# ------------------------------------------------- batch schedule ---------
NORMAL_T, SPIKE_T = 100.0, 80.0
D_NORMAL, D_SPIKE = 0.12, 0.85

# ------------------------------------------------- policy card ------------
POLICY_VERSION = 6
POLICY_FILE = "policy_card.json"
WARM_START = True
EPS_WARM = 0.15
SAVE_EVERY_BATCHES = 12

SORTER_ACT = ("red", "blue", "scrap")
SPEED_ACT = ("FAST", "SLOW")
TRUST_ACT = ("A", "B", "AVG")
MAINT_ACT = ("RUN", "CAL_A", "CAL_B", "CAL_BOTH")
OPP_ACT = ("SEIZE", "SKIP")
DISAGREE_NAME = {0: "agree ", 1: "A-high", 2: "B-high"}
HEALTH_NAME = {0: "ok ", 1: "mid", 2: "BAD"}

# ------------------------------------------------- UR5e DH model ----------
A  = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
D  = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
AL = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
TCP = 0.16
JNAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
R_DOWN = np.diag([-1.0, 1.0, -1.0])


def _dh(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,     sa,      ca,     d],
                     [0.0,    0.0,     0.0,   1.0]])


def fk_T(q):
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(q[i], D[i], A[i], AL[i])
    T = T.copy()
    T[:3, 3] += T[:3, 2] * TCP + BASE_XYZ
    return T


def solve_ik(q_seed, p_target, iters=100, lam=0.06):
    q = np.array(q_seed, dtype=float)
    d = 1e-4
    for _ in range(iters):
        T0 = fk_T(q)
        e_p = p_target - T0[:3, 3]
        e_r = np.zeros(3)
        for c in range(3):
            e_r += np.cross(T0[:3, c], R_DOWN[:, c])
        e = np.concatenate([e_p, 0.5 * e_r])
        if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 5e-2:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            q2 = q.copy()
            q2[i] += d
            T2 = fk_T(q2)
            e2_p = p_target - T2[:3, 3]
            e2_r = np.zeros(3)
            for c in range(3):
                e2_r += np.cross(T2[:3, c], R_DOWN[:, c])
            J[:, i] = (np.concatenate([e2_p, 0.5 * e2_r]) - e) / d
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
        q = np.clip(q + dq, -math.pi, math.pi)
    return q


def lambda_update(Q, E, s, a, r_seg, s2, alpha):
    """Accumulating-trace Q(lambda) at decision-point granularity."""
    E[s][a] += 1.0
    delta = r_seg + GAMMA_META * max(Q[s2]) - Q[s][a]
    for x in range(len(Q)):
        for u in range(len(Q[0])):
            Q[x][u] += alpha * delta * E[x][u]
            E[x][u] *= GAMMA_META * LAMBDA


def decay_traces(E):
    for x in range(len(E)):
        for u in range(len(E[0])):
            E[x][u] *= GAMMA_META * LAMBDA


class Ur5eSorter(Supervisor):

    def __init__(self):
        super(Ur5eSorter, self).__init__()
        self.ts = int(self.getBasicTimeStep())
        self.motors, self.sens = [], []
        for n in JNAMES:
            m = self.getDevice(n)
            m.setVelocity(1.5)
            self.motors.append(m)
            try:
                s = self.getDevice(n + "_sensor")
            except Exception:
                s = self.getDevice(n)
            s.enable(self.ts)
            self.sens.append(s)
        self.grip = self.getDevice("connector")
        self.grip.enablePresence(self.ts)
        self.me = self.getSelf()
        self.mast = None

        self.blocks = {}
        for d, (colour, half) in BLOCK_DEFS.items():
            node = self.getFromDef(d)
            self.blocks[d] = {"def": d,
                              "field": node.getField("translation"),
                              "node": node, "colour": colour, "half": half,
                              "held": False, "scored": False,
                              "waiting": False, "dwell": 0.0}

        self.Q = [[0.0] * 3 for _ in range(4)]
        self.metaQ = [[0.0, 0.0] for _ in range(6)]
        self.trustQ = [[0.0] * 3 for _ in range(3)]
        self.maintQ = [[0.0] * 4 for _ in range(18)]
        self.maintE = [[0.0] * 4 for _ in range(18)]
        self.oppQ = [[0.0, 0.0] for _ in range(3)]
        self.oppE = [[0.0, 0.0] for _ in range(3)]

        self.eps, self.picks, self.batches = EPS0, 0, 0
        self.meta_eps = META_EPS0
        self.stats = {"good": [0, 0], "bad": [0, 0]}

        self.raw_a = self.raw_b = PROV_THRESHOLD
        self.drift_a = self.drift_b = 0.0
        self.bucket, self.scrap_view = 0, 0.0
        self.mast_state = "ok"
        self.dab_ema = 0.0
        self.trust = 2
        self.readings, self.calibrated = [], False
        self.belt_v = FAST_V
        self.calA_cmd, self.calB_cmd = 0, 0
        self._cal_prev = False
        self.hold_until = 0.0
        self.seg = {"mt": 0.0, "op": 0.0}
        self.opp_prev = None
        self.batch = {"s_sp": 0, "a_sp": 0, "s_tr": 0, "a_tr": 0,
                      "s_mt": 0, "a_mt": 0, "r": 0.0, "n": 0}
        self.active, self.plan = None, []
        self.s_idx, self.action = 0, 0
        self.phase_prev = None
        self.warm = self.load_policy()

    # ----------------------------------------------------- policy card ---
    def config_fp(self):
        return {"version": POLICY_VERSION, "fast_v": FAST_V,
                "slow_v": SLOW_V, "cal_g_cost": CAL_G_COST,
                "cal_both_cost": CAL_BOTH_COST,
                "batch_picks": BATCH_PICKS, "drift_ok": DRIFT_OK,
                "drift_high": DRIFT_HIGH, "agree_thr": AGREE_THR}

    def policy_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            POLICY_FILE)

    def named_policy(self):
        pol = {}
        for ci, cn in enumerate(("red", "blue")):
            for h, hn in enumerate(("normal", "short")):
                pol["sorter %s/%s" % (cn, hn)] = \
                    SORTER_ACT[self.greedy(self.Q[ci * 2 + h])]
        for al in range(3):
            for sp, spn in ((0, "FAST"), (1, "SLOW")):
                pol["speed alarm=%d cur=%s" % (al, spn)] = \
                    SPEED_ACT[self.greedy(self.metaQ[al * 2 + sp])]
        for tr, trn in enumerate(("agree", "A-high", "B-high")):
            pol["trust %s" % trn] = TRUST_ACT[self.greedy(self.trustQ[tr])]
        for da in range(3):
            for db in range(3):
                for sp, spn in ((0, "FAST"), (1, "SLOW")):
                    s = (da * 3 + db) * 2 + sp
                    pol["maint fused=%s spare=%s %s"
                        % (HEALTH_NAME[da], HEALTH_NAME[db], spn)] = \
                        MAINT_ACT[self.greedy(self.maintQ[s])]
        for d in range(3):
            pol["opp sicker=%s" % HEALTH_NAME[d]] = \
                OPP_ACT[self.greedy(self.oppQ[d])]
        return pol

    def save_policy(self):
        card = {"version": POLICY_VERSION,
                "saved_at_sim_s": round(self.getTime(), 2),
                "picks": self.picks,
                "epsilon": {"sorter": round(self.eps, 3),
                            "meta": round(self.meta_eps, 3)},
                "config": self.config_fp(),
                "q": {"sorter": self.Q, "speed": self.metaQ,
                      "trust": self.trustQ, "maint": self.maintQ,
                      "opp": self.oppQ},
                "policy": self.named_policy()}
        try:
            with open(self.policy_path(), "w") as f:
                json.dump(card, f, indent=1)
            print("[card] policy card saved (%d picks, %d batches)"
                  % (self.picks, self.batches))
        except OSError as e:
            print("[card] save failed: %s" % e)

    def load_policy(self):
        if not WARM_START:
            return False
        try:
            with open(self.policy_path()) as f:
                card = json.load(f)
        except (OSError, ValueError):
            return False
        if card.get("version") != POLICY_VERSION \
                or card.get("config") != self.config_fp():
            print("[card] policy card found but version/config mismatch "
                  "- cold start")
            return False
        q = card.get("q", {})
        tables = (("sorter", self.Q), ("speed", self.metaQ),
                  ("trust", self.trustQ), ("maint", self.maintQ),
                  ("opp", self.oppQ))
        try:
            for key, table in tables:
                rows = q[key]
                if len(rows) != len(table) \
                        or any(len(r) != len(table[0]) for r in rows):
                    raise ValueError(key)
                for i in range(len(table)):
                    table[i] = [float(v) for v in rows[i]]
        except (KeyError, TypeError, ValueError):
            print("[card] policy card malformed - cold start")
            return False
        self.eps = EPS_WARM
        self.meta_eps = EPS_WARM
        print("[card] WARM START from policy card (%d picks of history)"
              % card.get("picks", -1))
        return True

    # ------------------------------------------------------ mailboxes ----
    def poll_mast(self):
        if self.mast is None:
            self.mast = self.getFromDef("QUALITY_MAST")
            if self.mast is None:
                return
        for tok in self.mast.getField("customData").getSFString().split():
            k, _, v = tok.partition("=")
            if k == "rawA":
                self.raw_a = float(v)
            elif k == "rawB":
                self.raw_b = float(v)
            elif k == "dAB":
                self.dab_ema += 0.25 * (float(v) - self.dab_ema)
            elif k == "bucket":
                self.bucket = int(v)
            elif k == "scrap":
                self.scrap_view = float(v)
            elif k == "driftA":
                self.drift_a = float(v)
            elif k == "driftB":
                self.drift_b = float(v)
            elif k == "state":
                self.mast_state = v

    def publish_speed(self):
        self.me.getField("customData").setSFString(
            "v=%.3f calA=%d calB=%d trust=%s"
            % (self.belt_v, self.calA_cmd, self.calB_cmd,
               TRUST_ACT[self.trust]))

    # ------------------------------------------------------- perception --
    def fused_raw(self):
        return (self.raw_a if self.trust == 0 else
                self.raw_b if self.trust == 1 else
                0.5 * (self.raw_a + self.raw_b))

    def fused_drift(self):
        return (self.drift_a if self.trust == 0 else
                self.drift_b if self.trust == 1 else
                0.5 * (self.drift_a + self.drift_b))

    def spare_drift(self):
        return (self.drift_b if self.trust == 0 else
                self.drift_a if self.trust == 1 else
                max(self.drift_a, self.drift_b))

    def drift_bucket(self, d):
        return 0 if d < DRIFT_OK else (2 if d >= DRIFT_HIGH else 1)

    def trust_state(self):
        if self.dab_ema > AGREE_THR:
            return 1
        if self.dab_ema < -AGREE_THR:
            return 2
        return 0

    def greedy(self, qs):
        m = max(qs)
        return random.choice([i for i, v in enumerate(qs)
                              if abs(v - m) < 1e-9])

    def speed_bit(self):
        return 0 if self.belt_v > 0.5 * (FAST_V + SLOW_V) else 1

    def meta_state(self):
        return self.bucket * 2 + self.speed_bit()

    def maint_state(self):
        return (self.drift_bucket(self.fused_drift()) * 3
                + self.drift_bucket(self.spare_drift())) * 2 + self.speed_bit()

    def opp_state(self):
        return self.drift_bucket(max(self.drift_a, self.drift_b))

    # ------------------------------------------------------ arm motion ---
    def q_measured(self):
        return np.array([s.getValue() for s in self.sens])

    def go_to(self, palm_xyz, tol=0.05, timeout=8.0):
        q = solve_ik(self.q_measured(), np.array(palm_xyz))
        for m, v in zip(self.motors, q):
            m.setPosition(v)
        t = 0
        while t < timeout * 1000:
            if self.step(self.ts) == -1:
                return False
            t += self.ts
            self.supervise()
            if max(abs(self.sens[i].getValue() - q[i])
                   for i in range(6)) < tol:
                return True
        print("  arm move timeout")
        return True

    # ----------------------------------------------------- height sense --
    def height_class(self, raw):
        v = raw + random.uniform(-0.002, 0.002)
        self.readings.append(v)
        if len(self.readings) > 80:
            self.readings.pop(0)
        if len(self.readings) < 8:
            return 0 if v < PROV_THRESHOLD else 1
        c1, c2 = min(self.readings), max(self.readings)
        for _ in range(10):
            a = [x for x in self.readings if abs(x - c1) <= abs(x - c2)]
            b = [x for x in self.readings if abs(x - c1) > abs(x - c2)]
            if not a or not b:
                break
            c1, c2 = sum(a) / len(a), sum(b) / len(b)
        if abs(c1 - c2) < 0.008:
            return 0 if v < PROV_THRESHOLD else 1
        thr = 0.5 * (c1 + c2)
        if not self.calibrated:
            self.calibrated = True
            print("[ur5e] gauge tracker calibrated: tall ~%.3f, short ~%.3f "
                  "(threshold %.3f)" % (min(c1, c2), max(c1, c2), thr))
        return 0 if v < thr else 1

    # ----------------------------------------------------------- belt ----
    def belt_step(self):
        act = [b for b in self.blocks.values()
               if not b["scored"] and not b["held"] and not b["waiting"]
               and b["field"].getSFVec3f()[0] < 0.09]
        if not act:
            return
        act.sort(key=lambda b: -b["field"].getSFVec3f()[0])
        dt = self.belt_v * self.ts / 1000.0
        limit = PICK_X
        for b in act:
            x, y, z = b["field"].getSFVec3f()
            target = min(x + dt, limit)
            if target > x:
                b["field"].setSFVec3f([target, y, z])
                limit = target - QUEUE_GAP
            else:
                limit = x - QUEUE_GAP

    def queue_head(self):
        act = [b for b in self.blocks.values()
               if not b["scored"] and not b["held"] and not b["waiting"]]
        if not act:
            return None
        head = max(act, key=lambda b: b["field"].getSFVec3f()[0])
        return head if head["field"].getSFVec3f()[0] >= PICK_X - 0.01 \
            else None

    # ------------------------------------------------- batch scheduling --
    def d_target(self):
        t = self.getTime() % (NORMAL_T + SPIKE_T)
        return D_NORMAL if t < NORMAL_T else D_SPIKE

    def to_stock(self, blk):
        i = list(self.blocks).index(blk["def"]) % len(STOCK)
        x, z = STOCK[i]
        blk["field"].setSFVec3f([x, blk["half"], z])
        blk["node"].resetPhysics()
        blk["waiting"], blk["scored"], blk["held"] = True, False, False

    def release_one(self):
        waiting = [b for b in self.blocks.values() if b["waiting"]]
        if not waiting:
            return
        want_bad = random.random() < self.d_target()
        pool = [b for b in waiting if (b["half"] < BAD_HALF) == want_bad]
        blk = random.choice(pool or waiting)
        blk["waiting"] = False
        blk["field"].setSFVec3f([START_X + random.uniform(-0.02, 0.02),
                                 0.10 + blk["half"],
                                 random.uniform(-0.03, 0.03)])
        blk["node"].resetPhysics()

    # ---------------------------------------------------------- picks ----
    def start_pick(self, blk):
        h = self.height_class(self.fused_raw())
        s = (0 if blk["colour"] == "red" else 1) * 2 + h
        a = random.randrange(3) if random.random() < self.eps \
            else self.greedy(self.Q[s])
        self.s_idx, self.action = s, a
        self.active = blk
        blk["held"] = True
        target = BIN[a] if a < 2 else SCRAP
        print("pick %3d: %s %-6s trust=%s -> action %s"
              % (self.picks + 1, blk["colour"], "short" if h else "normal",
                 TRUST_ACT[self.trust], SORTER_ACT[a]))
        self.plan = [("hover",), ("grasp",), ("lift",), ("travel", target),
                     ("release_at", target), ("release",), ("pause", 0.5),
                     ("park",)]

    def grasp(self):
        blk = self.active
        x, y, z = blk["field"].getSFVec3f()
        top = y + blk["half"]
        got = False
        for depth in (PICK_GAP, PICK_GAP - 0.012):
            if not self.go_to((x, top + depth, z), tol=0.04, timeout=6.0):
                break
            for _ in range(30):
                if self.step(self.ts) == -1:
                    return
                self.supervise()
                if self.grip.getPresence() == 1:
                    self.grip.lock()
                    got = True
                    break
            if got:
                break
        if not got:
            print("  grasp missed, part back to stock")
            blk["held"] = False
            self.active = None
            self.reward(self.s_idx, self.action, R_MISS, None, "miss")
            self.to_stock(blk)
            self.plan = [("park",)]

    def release_block(self):
        self.grip.unlock()
        self.pending = (self.active, self.s_idx, self.action)
        self.pending_t = self.getTime()
        self.active["held"] = False
        self.active = None

    # ---------------------------------------------------- grading --------
    def grade(self, colour, real_bad, bin_a):
        if bin_a == 2:
            return (1.0, "scrap OK", False) if real_bad \
                else (-1.0, "scrap WASTE", False)
        if real_bad:
            return (-2.0 + CONTAM_CHARGE, "CONTAMINATED", True)
        ci = 0 if colour == "red" else 1
        return (1.0, "OK", False) if bin_a == ci else (-1.0, "WRONG", False)

    def reward(self, s, a, r, real_bad, tag):
        self.Q[s][a] += ALPHA * (r - self.Q[s][a])
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)
        self.picks += 1
        if real_bad is not None:
            key = "bad" if real_bad else "good"
            self.stats[key][1] += 1
            if r > 0:
                self.stats[key][0] += 1
        print("  -> %-13s r=%+.1f | good %d/%d, defects scrapped %d/%d"
              % (tag, r, self.stats["good"][0], self.stats["good"][1],
                 self.stats["bad"][0], self.stats["bad"][1]))
        pr = r + (THROUGHPUT_BONUS if self.speed_bit() == 0 else 0.0)
        self.batch["r"] += pr
        self.batch["n"] += 1
        self.seg["mt"] += pr
        self.seg["op"] += pr
        if self.batch["n"] >= BATCH_PICKS:
            self.close_batch()

    # ------------------------------------------- batch close + planning --
    def close_batch(self):
        b = self.batch
        self.batches += 1
        mean = b["r"] / max(1, b["n"])

        lambda_update(self.maintQ, self.maintE, b["s_mt"], b["a_mt"],
                      self.seg["mt"], self.maint_state(), MAINT_ALPHA)
        decay_traces(self.maintE)
        decay_traces(self.oppE)

        self.metaQ[b["s_sp"]][b["a_sp"]] += META_ALPHA * \
            (mean - self.metaQ[b["s_sp"]][b["a_sp"]])
        self.trustQ[b["s_tr"]][b["a_tr"]] += TRUST_ALPHA * \
            (mean - self.trustQ[b["s_tr"]][b["a_tr"]])
        self.meta_eps = max(META_EPS_MIN, self.meta_eps * META_DECAY)

        if self.batches % 10 == 0:
            self.dump_q()
        if self.batches % SAVE_EVERY_BATCHES == 0:
            self.save_policy()
        self.plan_batch()

    def plan_batch(self):
        s_sp = self.meta_state()
        s_mt = self.maint_state()
        s_tr = self.trust_state()
        ex = random.random() < self.meta_eps
        a_sp = random.randrange(2) if ex else self.greedy(self.metaQ[s_sp])
        a_mt = random.randrange(4) if ex else self.greedy(self.maintQ[s_mt])
        a_tr = random.randrange(3) if ex else self.greedy(self.trustQ[s_tr])

        old_trust = self.trust
        self.trust = a_tr
        if a_tr != old_trust:
            self.readings = []            # fused stream changed: re-track
        self.belt_v = FAST_V if a_sp == 0 else SLOW_V
        self.calA_cmd = 1 if a_mt in (1, 3) else 0
        self.calB_cmd = 1 if a_mt in (2, 3) else 0
        if a_mt > 0:
            both = (a_mt == 3)
            cost = CAL_BOTH_COST if both else CAL_G_COST
            dur = T_CAL_BOTH if both else T_CAL_ONE
            self.seg["mt"] += cost
            self.hold_until = max(self.hold_until,
                                  self.getTime() + dur + 1.0)
            print("[maint] planned %s: line paused %.0f s (cost %.1f)"
                  % (MAINT_ACT[a_mt], dur, cost))
        self.publish_speed()
        self.batch = {"s_sp": s_sp, "a_sp": a_sp, "s_tr": s_tr,
                      "a_tr": a_tr, "s_mt": s_mt, "a_mt": a_mt,
                      "r": 0.0, "n": 0}
        print("[batch] alarm=%d scrap %2d%% | dAB %+.4f (%s) trust %s | "
              "fused %s spare %s | speed %s | maint %s | eps %.2f"
              % (self.bucket, int(100 * self.scrap_view), self.dab_ema,
                 DISAGREE_NAME[s_tr], TRUST_ACT[a_tr],
                 HEALTH_NAME[self.drift_bucket(self.fused_drift())],
                 HEALTH_NAME[self.drift_bucket(self.spare_drift())],
                 SPEED_ACT[a_sp], MAINT_ACT[a_mt], self.meta_eps))

    # ------------------------------------------ opportunistic decisions --
    def opp_event(self):
        """Fires when a contamination hold starts (one we didn't choose)."""
        if self.mast_state.startswith("cal") \
                or self.calA_cmd or self.calB_cmd:
            return
        s = self.opp_state()
        if self.opp_prev is not None:
            sp, ap = self.opp_prev
            lambda_update(self.oppQ, self.oppE, sp, ap,
                          self.seg["op"], s, OPP_ALPHA)
        a = random.randrange(2) if random.random() < self.meta_eps \
            else self.greedy(self.oppQ[s])
        self.opp_prev = (s, a)
        self.seg["op"] = 0.0
        if a == 1:
            gauge = "A" if self.drift_a >= self.drift_b else "B"
            self.seg["op"] += OPP_COST
            if gauge == "A":
                self.calA_cmd = 1
            else:
                self.calB_cmd = 1
            self.publish_speed()
            self.hold_until = max(self.hold_until,
                                  self.getTime() + T_CAL_ONE + 1.0)
            print("[opp] SEIZE hold: calibrating gauge %s (sicker, "
                  "cost %.1f)" % (gauge, OPP_COST))
        else:
            print("[opp] skip hold")

    # ---------------------------------------------------------- debug ----
    def dump_q(self):
        print("maint policy (fused x spare, per speed) from Q:")
        for da in range(3):
            for db in range(3):
                cells = [MAINT_ACT[self.greedy(
                    self.maintQ[(da * 3 + db) * 2 + sp])] for sp in (0, 1)]
                print("   fused=%s spare=%s  FAST->%-8s SLOW->%-8s"
                      % (HEALTH_NAME[da], HEALTH_NAME[db],
                         cells[0], cells[1]))
        print("trust policy from Q (rows: agree, A-high, B-high):")
        for name, row in zip(("agree ", "A-high", "B-high"), self.trustQ):
            print("   %s -> %s" % (name, TRUST_ACT[self.greedy(row)]))

    # --------------------------------------------- scoring / recycling ---
    def supervise(self):
        self.poll_mast()
        cal_now = self.mast_state.startswith("cal")
        if self._cal_prev and not cal_now:
            self.calA_cmd = self.calB_cmd = 0     # cal finished: clear latch
            self.publish_speed()
        self._cal_prev = cal_now
        now = self.getTime()
        if now >= self.hold_until:
            self.belt_step()
        targets = list(BIN.items()) + [(2, SCRAP)]
        for blk in self.blocks.values():
            if blk["scored"] or blk["held"] or blk["waiting"]:
                continue
            x, y, z = blk["field"].getSFVec3f()
            for a_bin, (bx, bz) in targets:
                if abs(x - bx) < SCORE_R and abs(z - bz) < SCORE_R \
                        and y < SCORE_Y:
                    blk["scored"] = True
                    blk["dwell"] = now + 1.5
                    if self.pending and self.pending[0] is blk:
                        _, s, a = self.pending
                        self.pending = None
                        real_bad = blk["half"] < BAD_HALF
                        r, tag, hold = self.grade(blk["colour"], real_bad,
                                                  a_bin)
                        self.reward(s, a, r, real_bad, tag)
                        if hold:
                            self.hold_until = max(self.hold_until,
                                                  now + HOLD_T)
                            print("  *** ALARM: contamination - line hold "
                                  "%.0f s ***" % HOLD_T)
                            self.opp_event()
                    break
        if self.pending and now - self.pending_t > 5.0:
            _, s, a = self.pending
            self.pending = None
            self.reward(s, a, R_MISS, None, "lost")
        for blk in self.blocks.values():
            if blk["scored"] and now >= blk["dwell"]:
                self.to_stock(blk)
        if sum(1 for b in self.blocks.values() if not b["waiting"]) < 3:
            self.release_one()

    # ------------------------------------------------------------ main ---
    def run(self):
        print("[ur5e] v6: per-gauge cals, lambda credit, opp stops, "
              "policy card (%s start)"
              % ("warm" if self.warm else "cold"))
        print("[ur5e] note: adversarial drift is set in the MAST "
              "(quality_mast.py ADVERSARY flag)")
        self.poll_mast()
        self.plan_batch()
        try:
            while self.step(self.ts) != -1:
                self.poll_mast()
                ph = 0 if self.getTime() % (NORMAL_T + SPIKE_T) < NORMAL_T \
                    else 1
                if ph != self.phase_prev:
                    self.phase_prev = ph
                    print("=== new batch: target defect rate %.0f%% ==="
                          % (100 * (D_NORMAL if ph == 0 else D_SPIKE)))
                self.supervise()
                if self.getTime() < self.hold_until \
                        or self.mast_state.startswith("cal"):
                    continue
                if self.plan:
                    step = self.plan.pop(0)
                    kind = step[0]
                    if kind == "hover":
                        b = self.active
                        x, y, z = b["field"].getSFVec3f()
                        self.go_to((x, y + b["half"] + PRE_GRASP, z))
                    elif kind == "grasp":
                        self.grasp()
                    elif kind == "lift":
                        b = self.active
                        x, y, z = b["field"].getSFVec3f()
                        self.go_to((x, y + 2 * b["half"] + PRE_GRASP + 0.03,
                                    z))
                    elif kind == "travel":
                        bx, bz = step[1]
                        self.go_to((bx, TRAVEL_Y, bz))
                    elif kind == "release_at":
                        bx, bz = step[1]
                        self.go_to((bx, RELEASE_Y, bz))
                    elif kind == "release":
                        self.release_block()
                    elif kind == "pause":
                        end = self.getTime() + step[1]
                        while self.getTime() < end \
                                and self.step(self.ts) != -1:
                            self.supervise()
                    elif kind == "park":
                        self.go_to(PARK)
                elif self.active is None:
                    head = self.queue_head()
                    if head is not None:
                        self.start_pick(head)
        finally:
            self.save_policy()


if __name__ == "__main__":
    Ur5eSorter().run()
