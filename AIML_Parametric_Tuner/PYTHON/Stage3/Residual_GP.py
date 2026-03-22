from time import time
import numpy as np
import json
import os
from scipy.optimize import minimize

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_1d(X, eps=1e-8):
    mu = np.mean(X)
    sigma = np.std(X) + eps
    return (X - mu) / sigma, mu, sigma

def denormalize_1d(Xn, mu, sigma):
    return Xn * sigma + mu

# ============================================================
# DISTANCE + KERNEL
# ============================================================

def compute_sqdist(X):
    X = X.reshape(-1, 1)
    return (X - X.T) ** 2

def build_covariance_from_sqdist(sqdist, l, sigma_f):
    return sigma_f**2 * np.exp(-0.5 * sqdist / (l**2))

# ============================================================
# GP CORE
# ============================================================

def gp_cholesky_and_alpha(sqdist, y, l, sigma_f, sigma_n):
    K = build_covariance_from_sqdist(sqdist, l, sigma_f)

    jitter = max(1e-6, 1e-4 * sigma_f**2)
    Ky = K + (sigma_n**2 + jitter) * np.eye(len(y))

    L = np.linalg.cholesky(Ky)
    y = y.reshape(-1, 1)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return L, alpha

# ============================================================
# GP PREDICTION
# ============================================================

def gp_predict(sqdist, X_train, y_train, x_test, l, sigma_f, sigma_n):
    L, alpha = gp_cholesky_and_alpha(sqdist, y_train, l, sigma_f, sigma_n)

    k_star = sigma_f**2 * np.exp(
        -0.5 * ((X_train - x_test) ** 2) / (l**2)
    ).reshape(-1, 1)

    mu_star = (k_star.T @ alpha).item()

    v = np.linalg.solve(L, k_star)
    k_xx = sigma_f**2
    sigma_star = max(k_xx - (v.T @ v).item(), 0.0)

    return mu_star, sigma_star

# ============================================================
# OPTIMIZATION
# ============================================================

def neg_log_marginal_likelihood(log_l, sqdist, y, sigma_f, sigma_n):
    l = np.exp(log_l[0])
    L, alpha = gp_cholesky_and_alpha(sqdist, y, l, sigma_f, sigma_n)

    term1 = 0.5 * (y.reshape(1, -1) @ alpha).item()
    term2 = np.sum(np.log(np.diag(L)))
    term3 = 0.5 * len(y) * np.log(2 * np.pi)

    return term1 + term2 + term3

def optimize_l(sqdist, y, sigma_f, sigma_n):
    res = minimize(
        neg_log_marginal_likelihood,
        x0=np.array([0.0]),
        args=(sqdist, y, sigma_f, sigma_n),
        method="L-BFGS-B",
        bounds=[(-2, 2)]
    )
    if not res.success:
        raise RuntimeError(res.message)

    return np.exp(res.x[0])

# ============================================================
# STAGE-3 INFERENCE
# ============================================================

def stage3_predict_all(LR_metrics, X_train, y_train, sqdist, hyperparams, norm_stats):
    results = {}

    for m in LR_metrics.keys():
        x_norm = (LR_metrics[m] - norm_stats[m]["X_mu"]) / norm_stats[m]["X_sigma"]

        mu_n, sigma_n = gp_predict(
            sqdist[m],
            X_train[m],
            y_train[m],
            x_norm,
            hyperparams[m]["l"],
            hyperparams[m]["sigma_f"],
            hyperparams[m]["sigma_n"]
        )

        delta_mean = denormalize_1d(mu_n, norm_stats[m]["y_mu"], norm_stats[m]["y_sigma"])
        delta_var = sigma_n * (norm_stats[m]["y_sigma"] ** 2)

        final_mean = LR_metrics[m] + delta_mean
        final_std = np.sqrt(delta_var)

        results[m] = {
            "LR_value": LR_metrics[m],
            "GP_delta_mean": float(delta_mean),
            "final_mean": float(final_mean),
            "final_std": float(final_std)
        }

    return results

# ============================================================
# MAIN
# ============================================================

def main():

    LR_path = r"C:\3rd Yr\Projects\AIML_Parametric_Tuner\DATA\regression_LR_output.json"
    d_path  = r"C:\3rd Yr\Projects\AIML_Parametric_Tuner\DATA\regression_deltas.json"

    base_dir = os.path.dirname(LR_path)
    OUT_path = os.path.join(base_dir, "stage3_full_output.json")

    with open(LR_path, "r") as f:
        LR = json.load(f)

    with open(d_path, "r") as f:
        DELTA = json.load(f)

    metrics = ["tpHL", "tpLH", "Pavg", "tr", "tf"]

    X_train, y_train, sqdist, norm_stats = {}, {}, {}, {}

    # --------------------------------------------------------
    # PREP
    # --------------------------------------------------------
    for m in metrics:
        Xn, Xm, Xs = normalize_1d(np.array(LR[m]))
        yn, ym, ys = normalize_1d(np.array(DELTA[f"{m}_delta"]))

        X_train[m] = Xn
        y_train[m] = yn
        sqdist[m] = compute_sqdist(Xn)

        norm_stats[m] = {
            "X_mu": Xm, "X_sigma": Xs,
            "y_mu": ym, "y_sigma": ys
        }

    # --------------------------------------------------------
    # TRAIN GP
    # --------------------------------------------------------
    hyperparams = {}

    for m in metrics:
        sigma_f = np.std(y_train[m])
        sigma_n = 0.025 * sigma_f

        l_opt = optimize_l(sqdist[m], y_train[m], sigma_f, sigma_n)

        hyperparams[m] = {
            "l": float(l_opt),
            "sigma_f": float(sigma_f),
            "sigma_n": float(sigma_n)
        }

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------
    N = len(LR[metrics[0]])
    predictions = []

    for i in range(N):
        LR_point = {m: LR[m][i] for m in metrics}

        pred = stage3_predict_all(
            LR_point,
            X_train,
            y_train,
            sqdist,
            hyperparams,
            norm_stats
        )

        predictions.append(pred)

    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------
    output = {
        "hyperparameters": hyperparams,
        "predictions": predictions
    }

    with open(OUT_path, "w") as f:
        json.dump(output, f, indent=4)

    print("Saved full Stage-3 output to:")
    print(OUT_path)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()