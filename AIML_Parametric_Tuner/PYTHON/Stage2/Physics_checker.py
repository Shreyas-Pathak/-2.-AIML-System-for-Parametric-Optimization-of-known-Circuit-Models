import json
import numpy as np
from pathlib import Path

# ============================================================
# FILE PATHS (EDIT IF NEEDED)
# ============================================================

DATA_DIR = Path(r"C:\3rd Yr\Projects\AIML_Parametric_Tuner\DATA")

JSON_PATHS = {
    "WN": DATA_DIR / "Wn_Sweep_S2_results.json",
    "WP": DATA_DIR / "Wp_Sweep_S2_results.json",
    "LW": DATA_DIR / "Lw_Sweep_S2_results.json",
    "WW": DATA_DIR / "Ww_Sweep_S2_results.json",
}

# ============================================================
# CONSTANTS
# ============================================================

EPS_DELAY = 1e-12
EPS_NUM = 1e-15

RC_LOW = 0.8
RC_HIGH = 1.3
RC_DEVICE_LIMIT = 0.6
RC_WIRE_LIMIT = 2.0
RC_EXTREME = 10.0

CAP_ERR_TOL = 0.35

REL_SLOPE_WARN = 0.05        # 5% relative change
VIOLATION_HYST = 0.20        # 20% opposite slope

# ============================================================
# VIOLATION CODES
# ============================================================

OUTPUT_NOT_RAIL = 1
EXCESSIVE_UNDERSHOOT = 2
NEGATIVE_DELAY = 3
PULSE_WIDTH = 4
NON_MONOTONIC = 5
PHYSICS_BREAK = 6
NUMERICAL = 7
JITTER = 8
NEAR_ZERO = 9
RC_EXTREME_V = 10
WIRE_LIMITED = 11

DIFF_VIOLATION = 12
DIFF_WARNING = 13

# ============================================================
# EXPECTED DIFFERENTIAL TRENDS
# ============================================================

EXPECTED_TRENDS = {
    "WN": {"tpHL": -1, "tpavg": -1, "tf": -1, "Reffn": -1},
    "WP": {"tpLH": -1, "tpavg": -1, "tr": -1, "Reffp": -1},
    "LW": {"tpHL": +1, "tpLH": +1, "tpavg": +1},
    "WW": {"tpHL": +1, "tpLH": +1, "tpavg": +1},
}

# ============================================================
# HELPERS
# ============================================================

def RC_delay(R, C):
    return R * C


def Cphysics(p):
    Cox = 6e-3
    Cdiff_per_W = 3e-16
    C0 = 1e-16
    C1 = 2e-16

    Wn, Wp = p["WN"], p["WP"]
    Lw, Ww = p["LW"], p["WW"]
    CL = p["CL"]

    return (
        Cox * (Wn + Wp) * Lw +
        Cdiff_per_W * (Wn + Wp) +
        (C0 + C1 * Ww) * Lw +
        CL
    )

# ============================================================
# FIRST-ORDER PHYSICS CHECKER
# ============================================================

def physics_check(sample):
    p, m, s = sample["params"], sample["metrics"], sample["stimulus"]

    violations = []
    regimes = set()
    fatal = False

    VDD = p["VDD"]

    tpHL, tpLH = m["tpHL"], m["tpLH"]
    tr, tf = m["tr"], m["tf"]
    Reffn, Reffp = m["Reffn"], m["Reffp"]
    Qout = m["Qout"]
    Vmax, Vmin = m["Vout_max"], m["Vout_min"]

    # ---------------- Output validity ----------------

    if Vmax < 0.9 * VDD or Vmin > 0.1 * VDD:
        violations.append([OUTPUT_NOT_RAIL, "Vout"])
        fatal = True

    if Vmax > 1.05 * VDD or Vmin < -0.05 * VDD:
        violations.append([EXCESSIVE_UNDERSHOOT, "Vout"])

    # ---------------- Delay sanity ----------------

    for name, val in {"tpHL": tpHL, "tpLH": tpLH, "tr": tr, "tf": tf}.items():
        if val < 0:
            if abs(val) < EPS_DELAY:
                violations.append([JITTER, name])
            else:
                violations.append([NEGATIVE_DELAY, name])
                fatal = True

    # ---------------- Pulse width ----------------

    if tr > s["Ton"]:
        violations.append([PULSE_WIDTH, "tr"])
        fatal = True

    if tf > s["Toff"]:
        violations.append([PULSE_WIDTH, "tf"])
        fatal = True

    # ---------------- Non-monotonic ----------------

    if abs(tpLH) > EPS_DELAY and tr > 1.5 * abs(tpLH):
        violations.append([NON_MONOTONIC, "tr"])

    if abs(tpHL) > EPS_DELAY and tf > 1.5 * abs(tpHL):
        violations.append([NON_MONOTONIC, "tf"])

    # ---------------- Capacitance consistency ----------------

    Cphy = Cphysics(p)
    Ceff = Qout / VDD

    if abs((Cphy - Ceff) / max(Cphy, EPS_NUM)) > CAP_ERR_TOL:
        violations.append([PHYSICS_BREAK, "Ctotal"])

    # ---------------- Regime classification ----------------

    rHL = tpHL / RC_delay(Reffp, Cphy)
    rLH = tpLH / RC_delay(Reffn, Cphy)

    for r in (rHL, rLH):
        if not np.isfinite(r) or r <= 0:
            regimes.add(2)
            continue

        if r < RC_DEVICE_LIMIT:
            regimes.add(1)
        elif RC_LOW <= r <= RC_HIGH:
            regimes.add(0)
        else:
            regimes.add(2)

        if r > RC_WIRE_LIMIT:
            violations.append([WIRE_LIMITED, "RC"])

        if r > RC_EXTREME:
            violations.append([RC_EXTREME_V, "RC"])
            fatal = True

    final_regime = (
        0 if regimes == {0} else
        1 if regimes == {1} else
        3 if regimes == {0, 1} else
        2
    )

    return {
        "validity": 0 if fatal else 1,
        "fatal": fatal,
        "regime": final_regime,
        "violations": violations,
        "differential_violations": [],
        "differential_warnings": [],
    }

# ============================================================
# DIFFERENTIAL CHECKER (FIXED)
# ============================================================

def differential_check(samples, swept_param):
    trends = EXPECTED_TRENDS.get(swept_param, {})
    samples = sorted(samples, key=lambda s: s["params"][swept_param])

    results = [{"differential_violations": [], "differential_warnings": []}]

    for i in range(1, len(samples)):
        prev, curr = samples[i - 1], samples[i]
        dv, dw = [], []

        if prev["physics"]["fatal"] or curr["physics"]["fatal"]:
            results.append({"differential_violations": [], "differential_warnings": []})
            continue

        dp = curr["params"][swept_param] - prev["params"][swept_param]
        if abs(dp) < EPS_NUM:
            results.append({"differential_violations": [], "differential_warnings": []})
            continue

        for metric, sign in trends.items():
            y1 = prev["metrics"].get(metric)
            y2 = curr["metrics"].get(metric)
            if y1 is None or y2 is None:
                continue

            dy = y2 - y1
            slope = dy / dp
            rel_change = abs(dy) / max(abs(y1), EPS_NUM)

            # Strong contradiction → violation
            if slope * sign < -VIOLATION_HYST * abs(slope):
                dv.append([DIFF_VIOLATION, f"{metric}_vs_{swept_param}"])

            # Weak or flat → warning
            elif rel_change < REL_SLOPE_WARN:
                dw.append([DIFF_WARNING, f"{metric}_vs_{swept_param}"])

        results.append({
            "differential_violations": dv,
            "differential_warnings": dw
        })

    return results

# ============================================================
# MAIN DRIVER
# ============================================================

def run_file(json_path: Path, swept_param: str):
    with open(json_path, "r") as f:
        samples = json.load(f)

    for s in samples:
        s["physics"] = physics_check(s)

    diff_results = differential_check(samples, swept_param)

    for s, d in zip(samples, diff_results):
        s["physics"]["differential_violations"] = d["differential_violations"]
        s["physics"]["differential_warnings"] = d["differential_warnings"]

    with open(json_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Processed {json_path.name}")

def main():
    for swept_param, path in JSON_PATHS.items():
        run_file(path, swept_param)
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()