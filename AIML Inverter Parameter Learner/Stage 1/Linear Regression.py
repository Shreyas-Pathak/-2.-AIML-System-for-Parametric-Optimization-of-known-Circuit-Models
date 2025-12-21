import json
import numpy as np

with open(r"C:\3rd Yr\Projects\AIML Inverter Parameter Learner\DATA\lhs_dataset.json", 'r') as file:
    data = json.load(file)

X = np.ones((120, 5))
tp = np.zeros((120, 1))
Pavg = np.zeros((120, 1))
tr = np.zeros((120, 1))
tf = np.zeros((120, 1))
Ctotal = np.zeros((120, 1))
Reffn = np.zeros((120, 1))
Reffp = np.zeros((120, 1))
for i in range(120):
    X[i][1] = data[i]["params"]["WN"]
    X[i][2] = data[i]["params"]["WP"]
    X[i][3] = data[i]["params"]["LW"]
    X[i][4] = data[i]["params"]["WW"]

Xraw = X.copy()
mu = X[:, 1:].mean(axis=0)
sigma = X[:, 1:].std(axis=0)
X[:, 1:] = (X[:, 1:] - mu) / sigma
XT = X.T

for i in range(120):
    tp[i][0] = (data[i]["metrics"]["tpavg: (tpHL+tpLH)/2"])
    Pavg[i][0] = (data[i]["metrics"]["pav: (-1)*(Vdd*Iavg)"])
    tr[i][0] = (data[i]["metrics"]["trnext: t90_r - t10_r"])
    tf[i][0] = (data[i]["metrics"]["tfnext: t10_f - t90_f"])
    Ctotal[i][0] = (data[i]["metrics"]["ctotal: Qout/VDD"])
    Reffn[i][0] = (data[i]["metrics"]["reffn: tfnext/(2.2*Ctotal)"])
    Reffp[i][0] = (data[i]["metrics"]["reffp: trnext/(2.2*Ctotal)"])

A = np.dot(XT, X)
B = np.linalg.inv(A)
beta_vector_tp = np.dot(np.dot(B, XT), tp)
beta_vector_Pavg = np.dot(np.dot(B, XT), Pavg)
beta_vector_tr = np.dot(np.dot(B, XT), tr)
beta_vector_tf = np.dot(np.dot(B, XT), tf)
beta_vector_Ctotal = np.dot(np.dot(B, XT), Ctotal)
beta_vector_Reffn = np.dot(np.dot(B, XT), Reffn)
beta_vector_Reffp = np.dot(np.dot(B, XT), Reffp)

print (beta_vector_Ctotal)

def Regression_Prediction(Wn, Wp, Lw, Ww):
    x = np.array([Wn, Wp, Lw, Ww])
    x = (x - mu) / sigma
    A = np.hstack(([1], x))
    tp, Pavg, tr, tf, Ctotal, Reffn, Reffp = np.dot(A, beta_vector_tp).item(), np.dot(A, beta_vector_Pavg).item(), np.dot(A, beta_vector_tr).item(), np.dot(A, beta_vector_tf).item(), np.dot(A, beta_vector_Ctotal).item(), np.dot(A, beta_vector_Reffn).item(), np.dot(A, beta_vector_Reffp).item()
    result = [tp, Pavg, tr, tf, Ctotal, Reffn, Reffp]
    return result

Delay_delta = []
Pavg_delta = []
Rise_Time_delta = []
Fall_Time_delta = []
Ctotal_delta = []
Reffn_delta = []
Reffp_delta = []
for i in range(120):
    a, b, c, d = Xraw[i][1], Xraw[i][2], Xraw[i][3], Xraw[i][4]
    x = Regression_Prediction(a, b, c, d)
    Delay_delta.append((x[0]-tp[i][0]).item())
    Pavg_delta.append((x[0]-Pavg[i][0]).item())
    Rise_Time_delta.append((x[0]-tr[i][0]).item())
    Fall_Time_delta.append((x[0]-tf[i][0]).item())
    Ctotal_delta.append((x[0]-Ctotal[i][0]).item())
    Reffn_delta.append((x[0]-Reffn[i][0]).item())
    Reffp_delta.append((x[0]-Reffp[i][0]).item())
