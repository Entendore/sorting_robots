"""sorting_robot.py — tabular Q-learning e-puck sorter (final version).

State   : 3x3 grid x colour(none/red/blue/green) x bump  -> 72 states
Actions : 0 forward, 1 turn left, 2 turn right
Rewards : +10 correct zone, -10 wrong zone, +0.2 seeing object, -0.02/step

Features:
  * COMPETITIVE flag: cooperative (both credited for every ball) or
    competitive (only the robot touching the ball is credited)
  * per-episode CSV logs (learning_log_<robot>.csv) for plot_learning.py
    and dashboard.py
  * federated Q-table sharing every 5 episodes (share_*.csv mailbox files)
  * greedy evaluation with success rate after TRAINING_EPISODES
  * Q-table persistence (qtable_<robot>.csv), resumable training
"""

import csv
import glob
import os
import random

from controller import Supervisor

# ------------------------------- tuning ---------------------------------
NO_COLOR, RED, BLUE, GREEN = 0, 1, 2, 3
COLOR_NAME = {NO_COLOR: "none", RED: "red", BLUE: "blue", GREEN: "green"}
N_COLORS = 4

COMPETITIVE = False          # True: only the touching robot is credited
SHARE = True                 # True: blend Q-tables with peer every 5 eps

GRID = 3
ARENA_HALF = 1.0
CELL = 2.0 * ARENA_HALF / GRID
N_STATES = GRID * GRID * N_COLORS * 2
N_ACTIONS = 3

ZONES = {
    RED:   (0.75, 0.0, 0.18),
    BLUE:  (-0.75, 0.0, 0.18),
    GREEN: (0.0, -0.75, 0.18),
}
BALL_DEFS = {
    RED:   ["RED_BALL_1", "RED_BALL_2", "RED_BALL_3"],
    BLUE:  ["BLUE_BALL_1", "BLUE_BALL_2", "BLUE_BALL_3"],
    GREEN: ["GREEN_BALL_1", "GREEN_BALL_2", "GREEN_BALL_3"],
}
ORANGE_DEF = "ORANGE_BALL_1"          # probe object: no zone, no reward

ALPHA = 0.3
GAMMA = 0.9
EPS_START, EPS_END, EPS_DECAY = 0.9, 0.05, 0.97
ACTION_STEPS = 20
MAX_EPISODE_STEPS = 180
TRAINING_EPISODES = 180
EVAL_EPISODES = 10

FWD_SPEED, TURN_SPEED = 3.0, 2.0
REWARD_STEP, REWARD_SEE = -0.02, 0.2
REWARD_GOOD, REWARD_BAD = 10.0, -10.0
BUMP_THRESHOLD = 600


class SortingLearner(Supervisor):

    def __init__(self):
        super(SortingLearner, self).__init__()
        self.ts = int(self.getBasicTimeStep())
        self.robot_name = self.getName()

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.stop()

        self.sensors = []
        for i in range(8):
            s = self.getDevice("ps%d" % i)
            s.enable(2 * self.ts)
            self.sensors.append(s)

        self.camera = self.getDevice("camera")
        if self.camera is None:
            raise RuntimeError("No camera: add an E-puckCamera to the slot")
        self.camera.enable(2 * self.ts)

        self.balls = {}
        for colour, names in BALL_DEFS.items():
            for n in names:
                node = self.getFromDef(n)
                if node is None:
                    raise RuntimeError("DEF %s not found in the world" % n)
                field = node.getField("translation")
                self.balls[n] = {"node": node, "field": field,
                                 "spawn": field.getSFVec3f(),
                                 "colour": colour, "done": False,
                                 "correct": False}
        self.orange = self.getFromDef(ORANGE_DEF)
        self.orange_field = self.orange.getField("translation") \
            if self.orange else None
        self.orange_spawn = self.orange_field.getSFVec3f() \
            if self.orange_field else None

        self.Q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]
        self.load_q_table()
        self.epsilon = EPS_START
        self.episode = 0
        self.learning = True

    # ------------------------------------------------------- helpers ----
    def stop(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def set_action(self, a):
        if a == 0:
            lv = rv = FWD_SPEED
        elif a == 1:
            lv, rv = -TURN_SPEED, TURN_SPEED
        else:
            lv, rv = TURN_SPEED, -TURN_SPEED
        self.left_motor.setVelocity(lv)
        self.right_motor.setVelocity(rv)

    def seen_colour(self):
        w, h = self.camera.getWidth(), self.camera.getHeight()
        data = self.camera.getImageArray()
        if not data:
            return NO_COLOR
        r = g = b = n = 0
        for y in range(h // 4, h, 2):
            row = data[y]
            for x in range(w // 4, 3 * w // 4, 2):
                p = row[x]
                r += p[0]; g += p[1]; b += p[2]; n += 1
        r, g, b = r / n, g / n, b / n
        if r > 90 and r - g > 35 and r - b > 35:
            return RED
        if b > 90 and b - r > 35 and b - g > 35:
            return BLUE
        if g > 90 and g - r > 35 and g - b > 35:
            return GREEN
        return NO_COLOR

    def get_state(self):
        x, _, z = self.getSelf().getField("translation").getSFVec3f()
        col = min(GRID - 1, max(0, int((x + ARENA_HALF) / CELL)))
        row = min(GRID - 1, max(0, int((z + ARENA_HALF) / CELL)))
        colour = self.seen_colour()
        bump = 1 if max(s.getValue() for s in self.sensors) > BUMP_THRESHOLD \
            else 0
        return (row * GRID + col) * (N_COLORS * 2) + colour * 2 + bump, colour

    def choose_action(self, s):
        if self.learning and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        row = self.Q[s]
        best = max(row)
        return random.choice([i for i, v in enumerate(row)
                              if abs(v - best) < 1e-9])

    def zone_events(self):
        r = 0.0
        me = self.getSelf().getField("translation").getSFVec3f()
        for info in self.balls.values():
            if info["done"]:
                continue
            x, _, z = info["field"].getSFVec3f()
            for colour, (zx, zz, half) in ZONES.items():
                if abs(x - zx) <= half and abs(z - zz) <= half:
                    info["done"] = True
                    info["correct"] = (colour == info["colour"])
                    if COMPETITIVE:
                        mine = (me[0] - x) ** 2 + (me[2] - z) ** 2 < 0.13 ** 2
                        if not mine:
                            break          # peer's sort — stay silent
                    if info["correct"]:
                        r += REWARD_GOOD
                    else:
                        r += REWARD_BAD
                    break
        return r, all(i["done"] for i in self.balls.values())

    def reset_episode(self):
        for info in self.balls.values():
            sx, y, sz = info["spawn"]
            info["field"].setSFVec3f([sx + random.uniform(-0.03, 0.03), y,
                                      sz + random.uniform(-0.03, 0.03)])
            info["node"].resetPhysics()
            info["done"] = False
            info["correct"] = False
        if self.orange_field is not None:
            ox, oy, oz = self.orange_spawn
            self.orange_field.setSFVec3f(
                [ox + random.uniform(-0.03, 0.03), oy,
                 oz + random.uniform(-0.03, 0.03)])
            self.orange.resetPhysics()

    def learn(self, s, a, r, s2, terminal):
        target = r if terminal else r + GAMMA * max(self.Q[s2])
        self.Q[s][a] += ALPHA * (target - self.Q[s][a])

    # -------------------------------------------------------- episode ----
    def run_episode(self, greedy=False):
        self.reset_episode()
        total = 0.0
        for _ in range(MAX_EPISODE_STEPS):
            s, _ = self.get_state()
            a = self.choose_action(s) if not greedy \
                else random.choice([i for i, v in enumerate(self.Q[s])
                                    if abs(v - max(self.Q[s])) < 1e-9])
            self.set_action(a)
            for _ in range(ACTION_STEPS):
                if self.step(2 * self.ts) == -1:
                    return None
            zone_r, done = self.zone_events()
            s2, colour2 = self.get_state()
            reward = REWARD_STEP + zone_r + \
                (REWARD_SEE if colour2 != NO_COLOR else 0.0)
            if self.learning and not greedy:
                self.learn(s, a, reward, s2, done)
            total += reward
            if done:
                break
        self.stop()
        ok = sum(1 for i in self.balls.values()
                 if i["done"] and i["correct"])
        bad = sum(1 for i in self.balls.values()
                  if i["done"] and not i["correct"])
        return total, ok, bad

    # ------------------------------------------------------------- io ----
    def folder(self):
        return os.path.dirname(os.path.abspath(__file__))

    def tag(self):
        return self.robot_name.replace(" ", "_")

    def q_file(self):
        return os.path.join(self.folder(), "qtable_%s.csv" % self.tag())

    def save_q_table(self):
        try:
            with open(self.q_file(), "w", newline="") as f:
                csv.writer(f).writerows(self.Q)
        except OSError as e:
            print("[%s] could not save Q-table: %s" % (self.robot_name, e))

    def load_q_table(self):
        try:
            with open(self.q_file()) as f:
                for i, row in enumerate(csv.reader(f)):
                    if i >= N_STATES:
                        break
                    for j in range(N_ACTIONS):
                        self.Q[i][j] = float(row[j])
            print("[%s] loaded saved Q-table" % self.robot_name)
        except (OSError, IndexError, ValueError):
            print("[%s] starting with a fresh (empty) Q-table"
                  % self.robot_name)

    def log_episode(self, total, ok, bad):
        path = os.path.join(self.folder(), "learning_log_%s.csv" % self.tag())
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["robot", "episode", "return",
                                "correct", "wrong", "epsilon"])
                w.writerow([self.robot_name, self.episode, "%.2f" % total,
                            ok, bad, "%.3f" % self.epsilon])
        except OSError as e:
            print("[%s] log write failed: %s" % (self.robot_name, e))

    def share_q(self):
        """Publish our Q-table and blend in peers' (simple federated avg)."""
        mine = os.path.join(self.folder(), "share_%s.csv" % self.tag())
        try:
            with open(mine, "w", newline="") as f:
                csv.writer(f).writerows(self.Q)
        except OSError:
            return
        for path in glob.glob(os.path.join(self.folder(), "share_*.csv")):
            if os.path.basename(path) == os.path.basename(mine):
                continue
            try:
                with open(path) as f:
                    rows = list(csv.reader(f))
                for i in range(min(N_STATES, len(rows))):
                    for j in range(N_ACTIONS):
                        self.Q[i][j] = 0.5 * (self.Q[i][j] + float(rows[i][j]))
            except (OSError, IndexError, ValueError):
                continue          # peer file mid-write — skip this round
        print("[%s] blended Q-tables with peers" % self.robot_name)

    # ------------------------------------------------------------ main ---
    def run(self):
        print("[%s] learning to sort... (%d states x %d actions)"
              % (self.robot_name, N_STATES, N_ACTIONS))
        n_balls = len(self.balls)
        while True:
            result = self.run_episode()
            if result is None:
                self.save_q_table()
                break
            total, ok, bad = result
            self.episode += 1
            print("[%s] episode %3d | return %7.2f | sorted %d correct, "
                  "%d wrong | eps %.2f"
                  % (self.robot_name, self.episode, total, ok, bad,
                     self.epsilon))
            self.log_episode(total, ok, bad)
            if self.episode % 10 == 0:
                self.save_q_table()
            if SHARE and self.episode % 5 == 0:
                self.share_q()
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
            if self.episode == TRAINING_EPISODES:
                self.learning = False
                self.epsilon = 0.0
                self.save_q_table()
                print("[%s] TRAINING DONE - evaluating greedy policy..."
                      % self.robot_name)
                scores = []
                for _ in range(EVAL_EPISODES):
                    res = self.run_episode(greedy=True)
                    if res is None:
                        break
                    scores.append(res[1])
                    print("[%s]   eval: %d/%d correct (return %.2f)"
                          % (self.robot_name, res[1], n_balls, res[0]))
                    self.log_episode(res[0], res[1], res[2])
                rate = 100.0 * sum(scores) / float(n_balls * max(1, len(scores)))
                print("[%s] greedy success rate: %.0f%% of balls sorted "
                      "correctly" % (self.robot_name, rate))


if __name__ == "__main__":
    SortingLearner().run()
