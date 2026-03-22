DELAY



POWER CONSUMED



POWER EFFICIENCY



TIME EFFICIENCY



SWEEP Vin vs Vout



PARASITIC CAPACITANCES



Types of change in parameters:

&nbsp;	1.) Physical Structure of wires: Lw, Ww

&nbsp;	

&nbsp;	2.) Physical Structure of Mosfets: Wn/p, L, WDifn/p, LDif, Nfn/p (Dif W\\L is dependent of W\\kL 	                                   of MOSFET)



&nbsp;	3.) Material: RShPoly/W/Dif, 



&nbsp;	4.) Power and Load: VDD, CL (very unlikely but could be used in some specific cases where 	                    effective trade-off MAY balance out to match new requirements.





Checkers:

&nbsp;	1.) tr, tf ( input slew )

&nbsp;	    Since tr = (4.4VDD x CL)/(Mup x Cox x (Wp/L) x (VDD - |Vtp|)^2) and 

&nbsp;	    tf = (4.4VDD x CL)/(Mun x Cox x (Wn/L) x (VDD - Vtn)^2), These can be used to check the 	    correctness of baseline physics wrt to changed params suggested by ML.

&nbsp;	    Note:

&nbsp;		for real inverter, tr = 2.2 x (Reffn + RSn  + Rw) x (Cload + Cdiff + Cwire + Coverlap)

&nbsp;		tr = 2.2 x (Reffp + RSp  + Rw) x (Cload + Cdiff + Cwire + Coverlap)

&nbsp;		Reff = VDD/Iavg (approx.)









Indicators:

&nbsp;	1.) Performance Indicator:

&nbsp;		a. Power vs parameter

&nbsp;		b. Delay vs parameter



&nbsp;	2.) Signal Integrity Indicator:

&nbsp;		a. VOH and VOL levels vs Param 

&nbsp;		b. o/p Swing vs param

&nbsp;		c. Logical Correctness of modified param circuit



&nbsp;	3.) Dynamic behaviour indicator/Checker:

&nbsp;		a. tr vs parameter

&nbsp;		b. tf vs parameter



&nbsp;	4.) Physics Explanation indicator:

&nbsp;		a. parasitics vs param

&nbsp;		b. Effective R and C vs parameter



&nbsp;	5.) Plausibility check:

&nbsp;		a. Scaling Law adherence

&nbsp;		b. Monotonicity checks





The system runs an iterative loop: an ML tuner proposes quick, intuition-driven parameter tweaks; the Physics AI evaluates suggestions, supplies corrections and uncertainty estimates based on learned physics (residual + PINN style), and the loop uses SPICE (or measurements) to validate and update both models. Over time the Physics AI becomes a progressively better predictive/causal model of “what actually happens,” while the ML tuner becomes a fast generator of candidate designs guided by that learned physics.





**alternate ( more explicit ) gp\_predict function:**

def gp\_predict(X\_train, y\_train, x\_test, l, sigma\_f, sigma\_n):

&nbsp;   k\_star = np.zeros((X\_train.shape\[0], 1))

&nbsp;   k\_double\_star = k\_calc(x\_test, x\_test, l, sigma\_f)

&nbsp;   for i in range(X\_train.shape\[0]):

&nbsp;       k\_star\[i, 0] = k\_calc(x\_test, X\_train\[i], l, sigma\_f)

&nbsp;   K = build\_covariance\_matrix(X\_train, l, sigma\_f)

&nbsp;   kT = k\_star.T

&nbsp;   K\_noise = K + sigma\_n\*\*2 \* np.eye(K.shape\[0]) + 1e-6 \* np.eye(K.shape\[0]) # for jitter

&nbsp;   L = np.linalg.cholesky(K\_noise)

&nbsp;   y\_train = y\_train.reshape(-1, 1)

&nbsp;   alpha = np.linalg.solve(L.T, np.linalg.solve(L, y\_train))

&nbsp;   mu\_star = kT @ alpha

&nbsp;   mu\_star = mu\_star.item()

&nbsp;   v = np.linalg.solve(L, k\_star)

&nbsp;   sigma\_star = k\_double\_star - (v.T @ v).item()

&nbsp;   sigma\_star = max(sigma\_star, 0.0)

&nbsp;   return mu\_star, sigma\_star



**iterative build\_covariance\_matrix:**

def build\_covariance\_matrix(X, l, sigma\_f):

&nbsp;   n = len(X)

&nbsp;   K = np.zeros((n, n))

&nbsp;   for i in range(n):

&nbsp;       for j in range(n):

&nbsp;           K\[i, j] = rbf\_kernel(X\[i], X\[j], l, sigma\_f)

&nbsp;   return K













