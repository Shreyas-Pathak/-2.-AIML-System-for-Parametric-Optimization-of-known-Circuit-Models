import subprocess
import json
import shutil
from pathlib import Path
import numpy as np

# ==========================================================
# USER CONFIGURATION
# ==========================================================

LTSPICE_EXE = r"C:\Users\Shreyas Pathak\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

PROJECT_ROOT = Path(r"C:\3rd Yr\Projects\AIML Inverter Parameter Learner")

TEMPLATE_NETLIST = PROJECT_ROOT / "SPICE/Testbench.net"

NETLIST_DIR = PROJECT_ROOT / "spice/runs/netlists"
LOG_DIR = PROJECT_ROOT / "spice/runs/logs"
DATA_FILE = PROJECT_ROOT / "data/lhs_dataset.json"

N_SAMPLES = 120

# ==========================================================
# PARAMETER RANGES (PHYSICAL)
# ==========================================================

PARAM_RANGES = {
    "WN": (4e-6, 50e-6),
    "WP": (2e-6, 50e-6),
    "LW": (10e-6, 100e-6),
    "WW": (0.5e-6, 2e-6),
    "NFN": (1, 4),
    "NFP": (1, 4),
}

# ==========================================================
# LHS SAMPLING
# ==========================================================

def latin_hypercube_sampling(ranges, n_samples):
    dim = len(ranges)
    lhs = np.zeros((n_samples, dim))
    for i in range(dim):
        perm = np.random.permutation(n_samples)
        lhs[:, i] = (perm + np.random.rand(n_samples)) / n_samples

    samples = []
    keys = list(ranges.keys())

    for i in range(n_samples):
        sample = {}
        for j, k in enumerate(keys):
            lo, hi = ranges[k]
            val = lo + lhs[i, j] * (hi - lo)
            if k.startswith("NF"):
                val = int(round(val))
            sample[k] = val
        samples.append(sample)

    return samples

# ==========================================================
# NETLIST GENERATION
# ==========================================================

def write_netlist(template_path, output_path, params):
    with open(template_path, "r") as f:
        text = f.read()

    for k, v in params.items():
        text = text.replace(f"__{k}__", str(v))

    with open(output_path, "w") as f:
        f.write(text)

# ==========================================================
# LOG PARSING
# ==========================================================

def parse_log(log_path):
    results = {}
    with open(log_path, "r") as f:
        for line in f:
            if "=" in line:
                parts = line.strip().split("=")
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        val = float(parts[1].strip())
                        results[key] = val
                    except ValueError:
                        pass
    return results

# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():

    for p in [TEMPLATE_NETLIST, LTSPICE_EXE]:
        if not Path(p).exists():
            raise RuntimeError(f"Missing required file: {p}")

    NETLIST_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    samples = latin_hypercube_sampling(PARAM_RANGES, N_SAMPLES)
    dataset = []

    for i, params in enumerate(samples, start=1):
        print(f"Running sample {i}/{N_SAMPLES}")

        netlist_path = NETLIST_DIR / f"run_{i:03d}.net"
        write_netlist(TEMPLATE_NETLIST, netlist_path, params)

        # ---- Run LTspice (Windows-safe) ----
        subprocess.run(
            f'"{LTSPICE_EXE}" -b "{netlist_path}"',
            shell=True,
            check=True
        )

        # ---- Locate log file ----
        raw_log = netlist_path.with_suffix(".log")
        if not raw_log.exists():
            print("  ❌ No log generated, skipping sample")
            continue

        final_log = LOG_DIR / raw_log.name
        shutil.move(raw_log, final_log)

        results = parse_log(final_log)
        if not results:
            print("  ⚠️ Empty measurements, skipping")
            continue

        dataset.append({
            "params": params,
            "metrics": results
        })

    with open(DATA_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✔ Dataset saved to {DATA_FILE}")

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
