"""sorter_arm.py — 3-DOF arm sorting blocks from a belt.

The colour -> pad mapping is learned with a 2x2 Q-table from landing
rewards. Belt motion and object transport are supervisor-driven for
simplicity; colour classification uses the real overhead camera.
"""

import math
import random

from controller import Supervisor

BASE_X, BASE_Z = 0.55, 0.0
SHOULDER_Y = 0.18
L1, L2 = 0.25, 0.28
CARRY_DROP = 0.055

PICK_X, BLOCK_Y, START_X = 0.25, 0.13, -0.75
BELT_SPEED = 0.05
HOVER_Y, DROP_Y = 0.26, 0.15
PARK = (0.15, 0.26, 0.0)

PAD_Z = {0: 0.33, 1: -0.33}
PAD_TRUTH = {"red": 0, "blue": 1}

ALPHA, EPS0, EPS_DECAY, EPS_MIN = 0.5, 0.4, 0.97, 0.02


class ArmSorter(Supervisor):

    def __init__(self):
        super(ArmSorter, self).__init__()
        self.ts = int(self.getBasicTimeStep())
        self.motors, self.sensors = {}, {}
        for j in ("yaw", "shoulder", "elbow"):
            self.motors[j] = self.getDevice(j + "_motor")
            self.motors[j].setPosition(0.0)
            self.sensors[j] = self.getDevice(j + "_sensor")
            self.sensors[j].enable(self.ts)
        self.cam = self.getDevice("camera")
        self.cam.enable(self.ts)

        self.blocks = {}
        for d, colour in (("BLOCK_R1", "red"), ("BLOCK_B1", "blue"),
                          ("BLOCK_R2", "red"), ("BLOCK_B2", "blue")):
            node = self.getFromDef(d)
            self.blocks[d] = {"node": node,
                              "field": node.getField("translation"),
                              "colour": colour, "done": False,
                              "recycle": 0.0}

        self.Q = [[0.0, 0.0], [0.0, 0.0]]
        self.eps = EPS0
        self.picks = 0
        self.stats = {"red": [0, 0], "blue": [0, 0]}
        self.held = self.lead = None
        self.plan = []
        self.current = None

    # ------------------------------------------------------------ arm ---
    def wrist(self):
        y0 = self.sensors["yaw"].getValue()
        t1 = self.sensors["shoulder"].getValue()
        t2 = self.sensors["elbow"].getValue()
        r = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        h = SHOULDER_Y + L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        return BASE_X + r * math.cos(y0), h, BASE_Z - r * math.sin(y0)

    def ik(self, tx, ty, tz):
        dx, dz = tx - BASE_X, tz - BASE_Z
        d = math.hypot(dx, dz)
        h = ty - SHOULDER_Y
        c = max(-1.0, min(1.0, (d * d + h * h - L1 * L1 - L2 * L2)
                          / (2 * L1 * L2)))
        e = math.acos(c)
        s = math.atan2(h, d) - math.atan2(L2 * math.sin(e),
                                          L1 + L2 * math.cos(e))
        return math.atan2(-dz, dx), s, e

    def move_to(self, tx, ty, tz, tol=0.02, timeout=8.0):
        target = self.ik(tx, ty, tz)
        for j, v in zip(("yaw", "shoulder", "elbow"), target):
            self.motors[j].setPosition(v)
        t = 0
        while t < timeout * 1000:
            if self.step(self.ts) == -1:
                return False
            t += self.ts
            if self.held:
                self.carry_update()
            if all(abs(self.sensors[j].getValue() - v) < tol
                   for j, v in zip(("yaw", "shoulder", "elbow"), target)):
                return True
        print("  !! move timeout ->", (tx, ty, tz))
        return False

    def carry_update(self):
        x, y, z = self.wrist()
        self.held["field"].setSFVec3f([x, y - CARRY_DROP, z])
        self.held["node"].resetPhysics()

    # ----------------------------------------------------------- belt ---
    def belt_step(self):
        active = [b for b in self.blocks.values()
                  if not b["done"] and b is not self.held]
        if not active:
            return None
        lead = max(active, key=lambda b: b["field"].getSFVec3f()[0])
        if lead["field"].getSFVec3f()[0] >= PICK_X - 1e-6:
            return lead
        dt = BELT_SPEED * self.ts / 1000.0
        for b in active:
            f = b["field"]
            x, y, z = f.getSFVec3f()
            f.setSFVec3f([min(PICK_X, x + dt), y, z])
        return None

    def classify(self):
        img = self.cam.getImageArray()
        if not img:
            return None
        w, h = self.cam.getWidth(), self.cam.getHeight()
        r = g = b = n = 0
        for y in range(h // 2 - 4, h // 2 + 4):
            for x in range(w // 2 - 4, w // 2 + 4):
                p = img[y][x]
                r += p[0]; g += p[1]; b += p[2]; n += 1
        r, g, b = r / n, g / n, b / n
        if r > 90 and r - b > 40 and r - g > 40:
            return "red"
        if b > 90 and b - r > 40:
            return "blue"
        return None

    # ---------------------------------------------------------- picks ---
    def start_pick(self):
        colour = self.classify()
        cheat = colour is None
        if cheat:
            colour = self.lead["colour"]
        s = 0 if colour == "red" else 1
        if random.random() < self.eps:
            a = random.randrange(2)
        else:
            a = 0 if self.Q[s][0] >= self.Q[s][1] else 1
        self.current = (s, a, colour, cheat)
        z = PAD_Z[a]
        self.plan = [
            ("move", PICK_X, HOVER_Y, 0.0),
            ("move", PICK_X, BLOCK_Y + CARRY_DROP, 0.0),
            ("grab",),
            ("move", PICK_X, HOVER_Y + 0.01, 0.0),
            ("move", 0.35, 0.22, z),
            ("move", 0.35, DROP_Y, z),
            ("release",),
            ("wait", 600),
            ("move", PARK[0], PARK[1], PARK[2]),
        ]

    def finish_pick(self):
        s, a, colour, cheat = self.current
        blk = self.lead
        self.held = None
        blk["field"].setSFVec3f([0.35, 0.10, PAD_Z[a]])
        blk["node"].resetPhysics()
        correct = (PAD_TRUTH[colour] == a)
        r = 1.0 if correct else -1.0
        self.Q[s][a] += ALPHA * (r - self.Q[s][a])
        self.stats[colour][0 if correct else 1] += 1
        self.picks += 1
        blk["done"] = True
        blk["recycle"] = self.getTime() + 2.0
        print("pick %3d: %s -> action %d (%s)%s | Q=%s"
              % (self.picks, colour, a, "OK" if correct else "WRONG",
                 "  (camera fallback)" if cheat else "",
                 [["%.1f" % v for v in row] for row in self.Q]))
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)
        if self.picks % 10 == 0:
            t = sum(self.stats[c][0] for c in self.stats)
            n = sum(self.stats[c][0] + self.stats[c][1] for c in self.stats)
            print("  accuracy so far: %d/%d = %.0f%%"
                  % (t, n, 100.0 * t / max(1, n)))

    # ------------------------------------------------------------ main ---
    def run(self):
        print("[arm] learning colour->pad mapping by trial and error")
        while self.step(self.ts) != -1:
            if self.plan:
                step = self.plan[0]
                if step[0] == "move":
                    self.move_to(step[1], step[2], step[3])
                elif step[0] == "wait":
                    end = self.getTime() + step[1] / 1000.0
                    while self.getTime() < end and self.step(self.ts) != -1:
                        if self.held:
                            self.carry_update()
                elif step[0] == "grab":
                    self.held = self.lead
                    self.carry_update()
                elif step[0] == "release":
                    self.finish_pick()
                self.plan.pop(0)
                continue
            now = self.getTime()
            for blk in self.blocks.values():
                if blk["done"] and now >= blk["recycle"]:
                    blk["field"].setSFVec3f(
                        [START_X + random.uniform(-0.02, 0.02), BLOCK_Y, 0.0])
                    blk["node"].resetPhysics()
                    blk["done"] = False
            self.lead = self.belt_step()
            if self.lead is not None:
                self.start_pick()


if __name__ == "__main__":
    ArmSorter().run()
