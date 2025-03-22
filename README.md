# === Sets ===
I = [1, 2, 3]               # Collection Centers
J = [1, 2]                  # Repacking Facilities
K = [1, 2, 3]               # Markets (not yet used in parameters but likely part of demand)
L = [1, 2]                  # Battery Types: 1=NMC, 2=LFP
T = [1, 2]                  # Time Periods
P = [1, 2]                  # Transport Modes: 1=Truck, 2=Rail
M = [1, 2, 3]               # Materials: 1=Li, 2=Co, 3=Al
R = [1, 2, 3]               # Recovery options: 1=Reuse, 2=Repurpose, 3=Recycle

# === Parameters ===
c = {(i, l, t): 5 + i + l + t for i in I for l in L for t in T}           # Collection cost
r = {(j, l, t): 30 + j + l + t for j in J for l in L for t in T}         # Repacking cost
t_cost = {(i, j, l, p, t): 2 + i + j + l + p + t for i in I for j in J for l in L for p in P for t in T}  # Transport cost
e = {(j, l, t): round(0.1 + 0.05 * l + 0.02 * j + 0.01 * t, 3) for j in J for l in L for t in T}  # Environmental impact
f = {(i, j, p): 0.5 if p == 1 else 0.3 for i in I for j in J for p in P}  # Emission factor
s = {(j, l): 10000 + 1000 * j + 500 * l for j in J for l in L}           # Setup cost
weights = {'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2}

Ca = {(i, t): 1000 + 200 * i + 100 * t for i in I for t in T}            # Collection capacity
Caj = {(j, t): 2000 + 300 * j + 100 * t for j in J for t in T}           # Processing capacity
SC = {j: 5000 + 500 * j for j in J}                                      # Storage capacity

RR = {1: 0.8, 2: 0.7}                # Recovery rate per battery type
Mr = 0.6                             # Min required recovery rate
Me = 100                             # Max emissions
Mw = 50                              # Max waste

SH = {(j, l, t): round(0.65 + 0.1 * j + 0.05 * l + 0.01 * t, 2) for j in J for l in L for t in T}  # State of health
Min_SH = 0.7

Mbz = 500                            # Max batch size for transport
Sf = 1.1                             # Safety factor
Mu = 0.5                             # Min utilization rate

Oc = {(j, t): 6000 + 500 * j + 200 * t for j in J for t in T}           # Operating cost
Ic = {j: 50000 + 10000 * j for j in J}                                  # Installation cost
Mb = {(j, t): 80000 + 5000 * j + 1000 * t for j in J for t in T}        # Budget

# === Example use ===
print("Collection cost at center 1 for NMC in period 1:", c[1, 1, 1])
print("Transport cost from center 1 to facility 1 for LFP via rail in period 2:", t_cost[1, 1, 2, 2, 2])
print("State of health for facility 2, NMC, period 1:", SH[2, 1, 1])
