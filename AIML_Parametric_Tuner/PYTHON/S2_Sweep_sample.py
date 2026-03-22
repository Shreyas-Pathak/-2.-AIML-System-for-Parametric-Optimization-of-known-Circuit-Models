from pathlib import Path
import json
import subprocess
import shutil
from typing import Dict, Any, List

# ============================================================
# IMPORT LTspice SAMPLER UTILITIES
# ============================================================
try:
    from LH_Sampler_inverter import *  # noqa
except Exception:
    try:
        from .LH_Sampler_inverter import *  # noqa
    except Exception as e:
        raise ImportError(
            "Failed to import LH_Sampler_inverter.py — check location"
        ) from e


# ============================================================
# FIXED PARAMETERS (non-swept defaults)
# ============================================================
BASE_PARAMS = {
    "WN": 2e-6,
    "WP": 4e-6,
    "LW": 1e-5,
    "WW": 1e-6,
    "CL": 5e-15,
    "VDD": 5
}

# stimulus
TON = 2e-10
TP = 5e-10
TOFF = TP - TON


# ============================================================
# NETLIST MOD HELPERS
# ============================================================
def delete_line_tbs2(file_path: Path, text_to_delete: str) -> None:
    if not file_path.exists():
        return
    target = text_to_delete.rstrip("\n")
    lines = file_path.read_text().splitlines()
    lines = [ln for ln in lines if ln.rstrip() != target]
    file_path.write_text("\n".join(lines) + "\n")


def append_line_to_testbenchS2(file_path: Path, line: str) -> None:
    lines = file_path.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(".backanno") or ln.strip().startswith(".end"):
            lines.insert(i, line)
            break
    else:
        lines.append(line)
    file_path.write_text("\n".join(lines) + "\n")


# ============================================================
# LTspice LOG PARSER
# ============================================================
def parse_ltspice_log(log_path: Path) -> List[Dict[str, Any]]:
    entries = []
    current = None

    for ln in log_path.read_text().splitlines():
        if ln.startswith(".step"):
            if current:
                entries.append(current)
            current = {"step": ln, "msgs": []}
        else:
            if current and ln.strip():
                current["msgs"].append(ln)

    if current:
        entries.append(current)

    return entries


# ============================================================
# METRIC EXTRACTION (CORRECTED)
# ============================================================
def extract_metrics(log_path: Path, param_name: str) -> List[Dict[str, Any]]:

    entries = parse_ltspice_log(log_path)
    pname = param_name.lower()

    step_values: List[float] = []
    for e in entries:
        step = e["step"].lower()
        step_values.append(float(step.split(f"{pname}=")[-1]))

    measurements: Dict[str, Dict[int, List[str]]] = {}
    current = None

    for e in entries:
        for ln in e["msgs"]:
            if ln.startswith("Measurement:"):
                current = ln.split("Measurement:")[1].split()[0].lower()
                measurements.setdefault(current, {})
            elif current and ln.split()[0].isdigit():
                idx = int(ln.split()[0])
                measurements[current][idx] = ln.split()[1:]

    samples: List[Dict[str, Any]] = []

    for i, pval in enumerate(step_values):
        idx = i + 1
        metrics: Dict[str, float] = {}

        def diff(meas):
            if meas in measurements and idx in measurements[meas]:
                r = measurements[meas][idx]
                return float(r[-1]) - float(r[-2])
            return None

        metrics["tpHL"] = diff("tphl")
        metrics["tpLH"] = diff("tplh")

        scalar_map = {
            "tpavg": "tpavg",
            "trnext": "tr",
            "tfnext": "tf",
            "ctotal": "Ctotal",
            "reffn": "Reffn",
            "reffp": "Reffp",
            "qout": "Qout",
            "pav": "Pavg",
            "voutmax": "Vout_max",
            "voutmin": "Vout_min"
        }

        for meas, out in scalar_map.items():
            if meas in measurements and idx in measurements[meas]:
                metrics[out] = float(measurements[meas][idx][0])

        params = dict(BASE_PARAMS)
        params[param_name.upper()] = pval

        assert params[param_name.upper()] == pval

        samples.append({
            "params": params,
            "metrics": metrics,
            "stimulus": {
                "Ton": TON,
                "Toff": TOFF
            }
        })

    return samples


# ============================================================
# GENERIC SWEEP RUNNER
# ============================================================
def run_sweep(
    param: str,
    step_line: str,
    outfile: str,
    netlist_path: Path,
    data_dir: Path
):

    delete_line_tbs2(netlist_path, step_line)
    append_line_to_testbenchS2(netlist_path, step_line)

    subprocess.run(
        f'"{LTSPICE_EXE}" -b "{netlist_path}"',
        shell=True,
        check=True
    )

    raw_log = netlist_path.with_suffix(".log")
    if not raw_log.exists():
        raise RuntimeError("LTspice log not generated")

    final_log = LOG_DIR / raw_log.name
    shutil.move(raw_log, final_log)

    samples = extract_metrics(final_log, param)

    with (data_dir / outfile).open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    delete_line_tbs2(netlist_path, step_line)


# ============================================================
# MAIN
# ============================================================
def main():

    project_root = Path(r"C:\3rd Yr\Projects\AIML_Parametric_Tuner")

    data_dir = project_root / "DATA"
    data_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = project_root / "SPICE" / "testbench_S2.net"

    append_line_to_testbenchS2(netlist_path, f".param CL={BASE_PARAMS['CL']}")
    append_line_to_testbenchS2(netlist_path, f".param VDD={BASE_PARAMS['VDD']}")
    run_sweep(
        "Wn",
        ".step param Wn 2u 50u 5u",
        "Wn_Sweep_S2_results.json",
        netlist_path,
        data_dir
    )

    run_sweep(
        "Wp",
        ".step param Wp 4u 50u 5u",
        "Wp_Sweep_S2_results.json",
        netlist_path,
        data_dir
    )

    run_sweep(
        "Lw",
        ".step param Lw 10u 100u 5u",
        "Lw_Sweep_S2_results.json",
        netlist_path,
        data_dir
    )

    run_sweep(
        "Ww",
        ".step param Ww 0.5u 2u 0.2u",
        "Ww_Sweep_S2_results.json",
        netlist_path,
        data_dir
    )

    delete_line_tbs2(netlist_path, f".param CL={BASE_PARAMS['CL']}")
    delete_line_tbs2(netlist_path, f".param VDD={BASE_PARAMS['VDD']}")

if __name__ == "__main__":
    main()
