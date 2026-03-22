DELAY

POWER CONSUMED

POWER EFFICIENCY

TIME EFFICIENCY

SWEEP Vin vs Vout

PARASITIC CAPACITANCES

Types of change in parameters:
	1.) Physical Structure of wires: Lw, Ww
	
	2.) Physical Structure of Mosfets: Wn/p, L, WDifn/p, LDif, Nfn/p (Dif W\L is dependent of W\kL 	                                   of MOSFET)

	3.) Material: RShPoly/W/Dif, 

	4.) Power and Load: VDD, CL (very unlikely but could be used in some specific cases where 	                    effective trade-off MAY balance out to match new requirements.


Checkers:
	1.) tr, tf ( input slew )
	    Since tr = (4.4VDD x CL)/(Mup x Cox x (Wp/L) x (VDD - |Vtp|)^2) and 
	    tf = (4.4VDD x CL)/(Mun x Cox x (Wn/L) x (VDD - Vtn)^2), These can be used to check the 	    correctness of baseline physics wrt to changed params suggested by ML.
	    Note:
		for real inverter, tr = 2.2 x (Reffn + RSn  + Rw) x (Cload + Cdiff + Cwire + Coverlap)
		tr = 2.2 x (Reffp + RSp  + Rw) x (Cload + Cdiff + Cwire + Coverlap)
		Reff = VDD/Iavg (approx.)




Indicators:
	1.) Performance Indicator:
		a. Power vs parameter
		b. Delay vs parameter

	2.) Signal Integrity Indicator:
		a. VOH and VOL levels vs Param 
		b. o/p Swing vs param
		c. Logical Correctness of modified param circuit

	3.) Dynamic behaviour indicator/Checker:
		a. tr vs parameter
		b. tf vs parameter

	4.) Physics Explanation indicator:
		a. parasitics vs param
		b. Effective R and C vs parameter

	5.) Plausibility check:
		a. Scaling Law adherence
		b. Monotonicity checks


The system runs an iterative loop: an ML tuner proposes quick, intuition-driven parameter tweaks; the Physics AI evaluates suggestions, supplies corrections and uncertainty estimates based on learned physics (residual + PINN style), and the loop uses SPICE (or measurements) to validate and update both models. Over time the Physics AI becomes a progressively better predictive/causal model of “what actually happens,” while the ML tuner becomes a fast generator of candidate designs guided by that learned physics.





