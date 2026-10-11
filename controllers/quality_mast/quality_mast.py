"""quality_mast.py — station PLC v5: DUAL height gauges + per-gauge
calibration + OPTIONAL adversarial drift.

Physics modelled here (none of it visible to the learner):
  * per gauge: vibration noise grows with belt speed
    (sigma = 0.0015 + NOISE_K * v), bias random-walks and accumulates
    ~speed-proportionally, drift widens noise.
  * ADVERSARY=True: the gauge the learner currently TRUSTS wears
    ADV_MULT x faster. The learner is never told; [adv] console lines are
    for the experimenter only. Set False for the benign baseline.
  * self-diagnosis: between parts the gauge must read AIR (0.5 m); a
    moving average of air readings departing 0.5 IS the drift symptom.
  * per-gauge calibration on rising-edge calA/calB commands: one gauge
    6 s, both 9 s (batch discount). Re-zeros only the commanded gauges.

The beacon classifies on the TRUSTED reading (plant-wide fusion policy),
so a wrong trust choice also degrades the alarm.

customData protocol (mast -> arm):
  rawA rawB dAB bucket scrap driftA driftB state
"""

import random

from controller import Robot

AIR = 0.5
BAND_LO, BAND_HI = 0.12, 0.45
FIRE_AT, REARM_STEPS = 15, 25
WINDOW = 12
Y_CAP, R_CAP = 0.25, 0.60
PROV_THRESHOLD = 0.28
NOISE_K = 0.19
DRIFT_WANDER = 8e-6
DRIFT_K = 1.0e-3
WIDEN_K = 0.8
T_CAL_ONE, T_CAL_BOTH = 6.0, 9.0
ADVERSARY = True                     # False = benign baseline
ADV_MULT = 3.5
ARM_DEF = "UR_ARM"


class Mast(Robot):

    def __init__(self):
        super(Mast, self).__init__()
        self.ts = int(self.getBasicTimeStep())
        self.dsa = self.getDevice("height_sensor_a")
        self.dsb = self.getDevice("height_sensor_b")
        self.dsa.enable(self.ts)
        self.dsb.enable(self.ts)
        self.lamp = [self.getDevice("beacon_green"),
                     self.getDevice("beacon_yellow"),
                     self.getDevice("beacon_red")]
        self.me = self.getSelf()
        self.arm = None
        self.speed = 0.06
        self.bias_a, self.bias_b = 0.0, 0.0
        self.air_ema_a, self.air_ema_b = AIR, AIR
        self.in_band, self.air, self.sa, self.sb = 0, 999, 0.0, 0.0
        self.readings, self.window = [], []
        self.bucket, self.scrap, self.dab_last = 0, 0.0, 0.0
        self.cal_until, self.cal_set = None, set()
        self.prev_a, self.prev_b = "0", "0"
        self.target, self.adv_prev = None, "unset"

    def classify(self, v):
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
        return 0 if v < 0.5 * (c1 + c2) else 1

    def set_lamps(self, k):
        for i, l in enumerate(self.lamp):
            l.set(1 if i == k else 0)

    def run(self):
        while self.step(self.ts) != -1:
            now = self.getTime()
            dt = self.ts / 1000.0
            req_a, req_b, trust_tok = "0", "0", "AVG"

            # ---- mailbox in: speed, per-gauge cal, trust directive -------
            if self.arm is None:
                self.arm = self.getFromDef(ARM_DEF)
            if self.arm is not None:
                for tok in self.arm.getField("customData") \
                        .getSFString().split():
                    k, _, v = tok.partition("=")
                    if k == "v":
                        try:
                            self.speed = float(v)
                        except ValueError:
                            pass
                    elif k == "calA":
                        req_a = v
                    elif k == "calB":
                        req_b = v
                    elif k == "trust":
                        trust_tok = v

            # adversary follows the learner's trust action
            self.target = trust_tok if trust_tok in ("A", "B") else None
            if ADVERSARY and self.target != self.adv_prev:
                if self.target:
                    print("[adv] gauge %s now TRUSTED -> wear x%.1f "
                          "(experimenter line)" % (self.target, ADV_MULT))
                else:
                    print("[adv] no gauge targeted (AVG)")
                self.adv_prev = self.target

            # per-gauge calibration trigger (rising edge, when idle)
            rising = (req_a == "1" and self.prev_a != "1") or \
                     (req_b == "1" and self.prev_b != "1")
            if rising and self.cal_until is None:
                self.cal_set = set()
                if req_a == "1":
                    self.cal_set.add("A")
                if req_b == "1":
                    self.cal_set.add("B")
                dur = T_CAL_BOTH if len(self.cal_set) == 2 else T_CAL_ONE
                self.cal_until = now + dur
                print("[mast] calibration stop (%s): line paused %.0f s"
                      % ("+".join(sorted(self.cal_set)), dur))
            self.prev_a, self.prev_b = req_a, req_b

            # ---- gauge wear (adversary multiplies the trusted gauge) -----
            ma = ADV_MULT if (ADVERSARY and self.target == "A") else 1.0
            mb = ADV_MULT if (ADVERSARY and self.target == "B") else 1.0
            self.bias_a += (random.gauss(0.0, DRIFT_WANDER)
                            + DRIFT_K * self.speed * dt) * ma
            self.bias_b += (random.gauss(0.0, DRIFT_WANDER)
                            + DRIFT_K * self.speed * dt) * mb

            # ---- calibration state machine --------------------------------
            in_cal = self.cal_until is not None
            if in_cal:
                blink = int(now * 4) % 2
                for i, l in enumerate(self.lamp):
                    l.set(1 if (i == 1 and blink) else 0)
                if now >= self.cal_until:
                    if "A" in self.cal_set:
                        self.bias_a, self.air_ema_a = 0.0, AIR
                    if "B" in self.cal_set:
                        self.bias_b, self.air_ema_b = 0.0, AIR
                    print("[mast] re-zeroed gauge(s) %s, sensing resumed"
                          % "+".join(sorted(self.cal_set)))
                    self.cal_set = set()
                    self.cal_until = None
                    self.in_band, self.sa, self.sb = 0, 0.0, 0.0
            sig = 0.0015 + NOISE_K * self.speed
            if in_cal:
                raw_a = raw_b = AIR
            else:
                raw_a = self.dsa.getValue() + self.bias_a \
                    + random.gauss(0.0, sig + WIDEN_K * abs(self.bias_a))
                raw_b = self.dsb.getValue() + self.bias_b \
                    + random.gauss(0.0, sig + WIDEN_K * abs(self.bias_b))

            # ---- air reference symptoms -----------------------------------
            self.air = 0 if raw_a < AIR else self.air + 1
            if raw_a >= AIR:
                self.air_ema_a += 0.02 * (raw_a - self.air_ema_a)
            if raw_b >= AIR:
                self.air_ema_b += 0.02 * (raw_b - self.air_ema_b)

            # ---- parked-part detection; beacon follows TRUSTED reading ----
            inb_a = BAND_LO < raw_a < BAND_HI
            inb_b = BAND_LO < raw_b < BAND_HI
            if in_cal:
                self.in_band = 0
            elif self.air >= REARM_STEPS and inb_a and inb_b:
                self.in_band += 1
                self.sa += raw_a
                self.sb += raw_b
                if self.in_band == FIRE_AT:
                    mean_a, mean_b = self.sa / FIRE_AT, self.sb / FIRE_AT
                    self.dab_last = mean_a - mean_b
                    fused = mean_a if trust_tok == "A" else \
                        mean_b if trust_tok == "B" else \
                        0.5 * (mean_a + mean_b)
                    self.window.append(self.classify(fused))
                    if len(self.window) > WINDOW:
                        self.window.pop(0)
                    self.scrap = sum(self.window) / float(len(self.window))
                    self.bucket = 2 if self.scrap >= R_CAP else \
                        (1 if self.scrap >= Y_CAP else 0)
                    self.set_lamps(self.bucket)
            elif not (inb_a and inb_b):
                self.in_band = 0
                self.sa, self.sb = 0.0, 0.0

            # ---- mailbox out ----------------------------------------------
            state = ("cal" + "".join(sorted(self.cal_set))) \
                if in_cal else "ok"
            self.me.getField("customData").setSFString(
                "rawA=%.4f rawB=%.4f dAB=%.4f bucket=%d scrap=%.2f "
                "driftA=%.4f driftB=%.4f state=%s"
                % (raw_a, raw_b, self.dab_last, self.bucket, self.scrap,
                   abs(self.air_ema_a - AIR), abs(self.air_ema_b - AIR),
                   state))


if __name__ == "__main__":
    Mast().run()
