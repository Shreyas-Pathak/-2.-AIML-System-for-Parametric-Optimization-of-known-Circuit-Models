import json
import numpy as np
from pathlib import Path

# ============================================================
# LOAD DATA
# ============================================================
input_path = Path(r"C:\3rd Yr\Projects\AIML_Parametric_Tuner\DATA\lhs_dataset.json")

with open(input_path, 'r') as file:
    data = json.load(file)

# ============================================================
# ALLOCATE ARRAYS
# ============================================================
N = 120

X = np.ones((N, 5))
tpHL = np.zeros((N, 1))
tpLH = np.zeros((N, 1))
Pavg = np.zeros((N, 1))
tr = np.zeros((N, 1))
tf = np.zeros((N, 1))
Ctotal = np.zeros((N, 1))
Reffn = np.zeros((N, 1))
Reffp = np.zeros((N, 1))

# ============================================================
# DESIGN MATRIX
# ============================================================
for i in range(N):
    X[i][1] = data[i]["params"]["WN"]
    X[i][2] = data[i]["params"]["WP"]
    X[i][3] = data[i]["params"]["LW"]
    X[i][4] = data[i]["params"]["WW"]

Xraw = X.copy()

mu = X[:, 1:].mean(axis=0)
sigma = X[:, 1:].std(axis=0)
X[:, 1:] = (X[:, 1:] - mu) / sigma

XT = X.T

# ============================================================
# METRICS
# ============================================================
for i in range(N):
    tpHL[i][0] = data[i]["metrics"]["tpHL"]
    tpLH[i][0] = data[i]["metrics"]["tpLH"]
    Pavg[i][0] = data[i]["metrics"]["pav: (-1)*(vdd*iavg)"]
    tr[i][0] = data[i]["metrics"]["trnext: t90_r - t10_r"]
    tf[i][0] = data[i]["metrics"]["tfnext: t10_f - t90_f"]
    Ctotal[i][0] = data[i]["metrics"]["ctotal: qout/vdd"]
    Reffn[i][0] = data[i]["metrics"]["reffn: tfnext/(2.2*ctotal)"]
    Reffp[i][0] = data[i]["metrics"]["reffp: trnext/(2.2*ctotal)"]

# ============================================================
# NORMAL EQUATION
# ============================================================
A = XT @ X
B = np.linalg.inv(A)

beta_vector_tpHL = B @ XT @ tpHL
beta_vector_tpLH = B @ XT @ tpLH
beta_vector_Pavg = B @ XT @ Pavg
beta_vector_tr = B @ XT @ tr
beta_vector_tf = B @ XT @ tf
beta_vector_Ctotal = B @ XT @ Ctotal
beta_vector_Reffn = B @ XT @ Reffn
beta_vector_Reffp = B @ XT @ Reffp

# ============================================================
# REGRESSION PREDICTOR
# ============================================================
def Regression_Prediction(Wn, Wp, Lw, Ww):
    x = np.array([Wn, Wp, Lw, Ww])
    x = (x - mu) / sigma
    A = np.hstack(([1], x))

    return [
        (A @ beta_vector_tpHL).item(),
        (A @ beta_vector_tpLH).item(),
        (A @ beta_vector_Pavg).item(),
        (A @ beta_vector_tr).item(),
        (A @ beta_vector_tf).item(),
        (A @ beta_vector_Ctotal).item(),
        (A @ beta_vector_Reffn).item(),
        (A @ beta_vector_Reffp).item()
    ]

# ============================================================
# DELTAS (FIXED)
# ============================================================
tpHL_delta = []
tpLH_delta = []
Pavg_delta = []
tr_delta = []
tf_delta = []
Ctotal_delta = []
Reffn_delta = []
Reffp_delta = []

# predicted (LR) outputs
tpHL_pred = []
tpLH_pred = []
Pavg_pred = []
tr_pred = []
tf_pred = []
Ctotal_pred = []
Reffn_pred = []
Reffp_pred = []

LR_output = input_path.parent / "regression_LR_output.json"

for i in range(N):
    a, b, c, d = Xraw[i][1], Xraw[i][2], Xraw[i][3], Xraw[i][4]
    pred = Regression_Prediction(a, b, c, d)
    # store predictions
    tpHL_pred.append(pred[0])
    tpLH_pred.append(pred[1])
    Pavg_pred.append(pred[2])
    tr_pred.append(pred[3])
    tf_pred.append(pred[4])
    Ctotal_pred.append(pred[5])
    Reffn_pred.append(pred[6])
    Reffp_pred.append(pred[7])

    # store deltas
    tpHL_delta.append(-pred[0] + tpHL[i][0])
    tpLH_delta.append(-pred[1] + tpLH[i][0])
    Pavg_delta.append(-pred[2] + Pavg[i][0])
    tr_delta.append(-pred[3] + tr[i][0])
    tf_delta.append(-pred[4] + tf[i][0])
    Ctotal_delta.append(-pred[5] + Ctotal[i][0])
    Reffn_delta.append(-pred[6] + Reffn[i][0])
    Reffp_delta.append(-pred[7] + Reffp[i][0])



# ============================================================
# SAVE DELTAS TO JSON
# ============================================================
output_delta = {
    "tpHL_delta": [float(x) for x in tpHL_delta],
    "tpLH_delta": [float(x) for x in tpLH_delta],
    "Pavg_delta": [float(x) for x in Pavg_delta],
    "tr_delta": [float(x) for x in tr_delta],
    "tf_delta": [float(x) for x in tf_delta],
    "Ctotal_delta": [float(x) for x in Ctotal_delta],
    "Reffn_delta": [float(x) for x in Reffn_delta],
    "Reffp_delta": [float(x) for x in Reffp_delta]
}

output_path = input_path.parent / "regression_deltas.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_delta, f, indent=4)

# ============================================================
# SAVE LR PREDICTIONS TO JSON
# ============================================================
output_pred = {
    "tpHL": [float(x) for x in tpHL_pred],
    "tpLH": [float(x) for x in tpLH_pred],
    "Pavg": [float(x) for x in Pavg_pred],
    "tr": [float(x) for x in tr_pred],
    "tf": [float(x) for x in tf_pred],
    "Ctotal": [float(x) for x in Ctotal_pred],
    "Reffn": [float(x) for x in Reffn_pred],
    "Reffp": [float(x) for x in Reffp_pred]
}

with open(LR_output, "w", encoding="utf-8") as f:
    json.dump(output_pred, f, indent=4)

