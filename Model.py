# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:09:30 2025

@author: hneko
"""

import pyomo.environ as pyo
import pandas as pd
import numpy as np

# Create a concrete model
model = pyo.ConcreteModel()

# Define sets
# For a realistic example, I'll use smaller set sizes
I = range(3)  # Collection centers
J = range(2)  # Repacking facilities
L = range(2)  # Battery types (e.g., EV, consumer electronics)
T = range(3)  # Time periods (e.g., quarters)
P = range(2)  # Transportation modes (e.g., truck, rail)
M = range(3)  # Materials (e.g., lithium, cobalt, nickel)
R = range(3)  # Recovery methods (direct reuse, repurpose, recycle)

# Add sets to the model
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)
model.L = pyo.Set(initialize=L)
model.T = pyo.Set(initialize=T)
model.P = pyo.Set(initialize=P)
model.M = pyo.Set(initialize=M)
model.R = pyo.Set(initialize=R)

# Define parameters (using random data for demonstration)
np.random.seed(42)  # For reproducibility

# Collection costs
collection_cost = {(i, l, t): np.random.uniform(10, 20) 
                  for i in I for l in L for t in T}
model.c = pyo.Param(model.I, model.L, model.T, initialize=collection_cost)

# Repacking costs
repacking_cost = {(j, l, t): np.random.uniform(15, 30) 
                 for j in J for l in L for t in T}
model.r = pyo.Param(model.J, model.L, model.T, initialize=repacking_cost)

# Transportation costs
transport_cost = {(i, j, l, p, t): np.random.uniform(5, 15) * (1 + 0.5 * p)  # Higher cost for better transport mode
                 for i in I for j in J for l in L for p in P for t in T}
model.t_cost = pyo.Param(model.I, model.J, model.L, model.P, model.T, initialize=transport_cost)

# Environmental impact
env_impact = {(j, l, t): np.random.uniform(1, 5) 
             for j in J for l in L for t in T}
model.e = pyo.Param(model.J, model.L, model.T, initialize=env_impact)

# Emission factors for transportation
emission_factor = {(i, j, p): np.random.uniform(0.5, 2) * (1 - 0.3 * p)  # Lower emissions for better transport mode
                  for i in I for j in J for p in P}
model.f = pyo.Param(model.I, model.J, model.P, initialize=emission_factor)

# Setup costs - reduced to address potential budget constraints
setup_cost = {(j, l): np.random.uniform(3000, 6000) 
             for j in J for l in L}
model.s = pyo.Param(model.J, model.L, initialize=setup_cost)

# Weight factors for objective function - adjusted to promote processing
model.alpha = pyo.Param(initialize=0.4)  # Cost weight
model.beta = pyo.Param(initialize=0.2)   # Environmental weight
model.gamma = pyo.Param(initialize=0.4)  # Circularity weight (increased)

# Collection capacity
collection_capacity = {(i, t): np.random.uniform(800, 1200) 
                      for i in I for t in T}
model.Ca_i = pyo.Param(model.I, model.T, initialize=collection_capacity)

# Processing capacity - increased to handle more volume
processing_capacity = {(j, t): np.random.uniform(2000, 3000) 
                      for j in J for t in T}
model.Ca_j = pyo.Param(model.J, model.T, initialize=processing_capacity)

# Storage capacity - increased
storage_capacity = {j: np.random.uniform(3000, 4000) for j in J}
model.SC = pyo.Param(model.J, initialize=storage_capacity)

# Expected recovery rate
recovery_rate = {l: np.random.uniform(0.6, 0.9) for l in L}
model.Rt = pyo.Param(model.L, initialize=recovery_rate)

# State of health
soh = {(j, l, t): np.random.uniform(0.7, 0.95) 
      for j in J for l in L for t in T}
model.SH = pyo.Param(model.J, model.L, model.T, initialize=soh)

# Quality threshold for each recovery method - relaxed thresholds
quality_threshold = {r: max(0.2, 1 - (r * 0.3)) for r in R}  # More relaxed threshold
model.QT = pyo.Param(model.R, initialize=quality_threshold)

# Recovery efficiency for each material and recovery method
recovery_efficiency = {(m, r): np.random.uniform(0.5, 0.95) * (1 - r * 0.1) 
                      for m in M for r in R}
model.RE = pyo.Param(model.M, model.R, initialize=recovery_efficiency)

# Material content in each battery type
material_content = {(m, l): np.random.uniform(0.05, 0.3) 
                   for m in M for l in L}
model.MC = pyo.Param(model.M, model.L, initialize=material_content)

# Demand
demand = {(l, t): np.random.uniform(500, 800) 
         for l in L for t in T}
model.demand = pyo.Param(model.L, model.T, initialize=demand)

# Maximum allowed emissions - increased
model.Me = pyo.Param(initialize=10000)

# Total budget - increased
model.TB = pyo.Param(initialize=1000000)

# Operating costs
op_cost = {(j, t): np.random.uniform(5000, 8000) 
          for j in J for t in T}
model.Oc = pyo.Param(model.J, model.T, initialize=op_cost)

# Installation costs
inst_cost = {j: np.random.uniform(50000, 80000) 
            for j in J}
model.Ic = pyo.Param(model.J, initialize=inst_cost)

# Decision Variables

# Amount collected
model.x = pyo.Var(model.I, model.L, model.T, domain=pyo.NonNegativeReals)

# Amount processed
model.y = pyo.Var(model.J, model.L, model.T, domain=pyo.NonNegativeReals)

# Amount processed by recovery method
model.y_r = pyo.Var(model.J, model.L, model.T, model.R, domain=pyo.NonNegativeReals)

# Amount transported
model.z = pyo.Var(model.I, model.J, model.L, model.P, model.T, domain=pyo.NonNegativeReals)

# Binary setup variable
model.b = pyo.Var(model.J, model.L, domain=pyo.Binary)

# Inventory
model.In = pyo.Var(model.J, model.L, model.T, domain=pyo.NonNegativeReals)

# Material recovered
model.Rm = pyo.Var(model.M, model.J, model.L, model.T, domain=pyo.NonNegativeReals)

# Quantity shipped out
model.Sh = pyo.Var(model.J, model.L, model.T, domain=pyo.NonNegativeReals)

# Add slack variables for debugging infeasibility
model.slack_demand = pyo.Var(model.L, model.T, domain=pyo.NonNegativeReals)
model.slack_process = pyo.Var(domain=pyo.NonNegativeReals)
model.slack_quality = pyo.Var(model.J, model.L, model.T, model.R, domain=pyo.NonNegativeReals)

# Objective Function (Linear version with slack penalties)
def obj_rule(model):
    # Economic component
    cost_term = sum(model.c[i, l, t] * model.x[i, l, t] for i in model.I for l in model.L for t in model.T) + \
                sum(model.r[j, l, t] * model.y[j, l, t] for j in model.J for l in model.L for t in model.T) + \
                sum(model.t_cost[i, j, l, p, t] * model.z[i, j, l, p, t] 
                    for i in model.I for j in model.J for l in model.L for p in model.P for t in model.T)
    
    # Environmental component
    env_term = sum(model.e[j, l, t] * model.y[j, l, t] for j in model.J for l in model.L for t in model.T) + \
               sum(model.f[i, j, p] * sum(model.z[i, j, l, p, t] for l in model.L for t in model.T) 
                   for i in model.I for j in model.J for p in model.P)
    
    # Simplified circularity component (linear version)
    # Instead of using ratios, we can maximize the absolute amount of recovered material
    circularity_term = sum(model.Rm[m, j, l, t] for m in model.M for j in model.J for l in model.L for t in model.T)
    
    # Slack penalty (high penalty for using slack variables)
    slack_penalty = 10000 * (sum(model.slack_demand[l, t] for l in model.L for t in model.T) + 
                            model.slack_process + 
                            sum(model.slack_quality[j, l, t, r] for j in model.J for l in model.L for t in model.T for r in model.R))
    
    return model.alpha * cost_term + model.beta * env_term - model.gamma * circularity_term + slack_penalty

model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Constraints

# Demand satisfaction with slack
def demand_rule(model, l, t):
    return sum(model.x[i, l, t] for i in model.I) + model.slack_demand[l, t] >= model.demand[l, t]

model.demand_constraint = pyo.Constraint(model.L, model.T, rule=demand_rule)

# Flow conservation
def flow_rule(model, j, l, t):
    return sum(model.z[i, j, l, p, t] for i in model.I for p in model.P) == model.y[j, l, t]

model.flow_constraint = pyo.Constraint(model.J, model.L, model.T, rule=flow_rule)

# Ensure most collected batteries are processed (with slack)
def process_collected_rule(model):
    return sum(model.y[j, l, t] for j in model.J for l in model.L for t in model.T) + model.slack_process >= \
           0.8 * sum(model.x[i, l, t] for i in model.I for l in model.L for t in model.T)

model.process_collected_constraint = pyo.Constraint(rule=process_collected_rule)

# Collection capacity
def coll_capacity_rule(model, i, t):
    return sum(model.x[i, l, t] for l in model.L) <= model.Ca_i[i, t]

model.coll_capacity_constraint = pyo.Constraint(model.I, model.T, rule=coll_capacity_rule)

# Processing capacity
def proc_capacity_rule(model, j, t):
    return sum(model.y[j, l, t] for l in model.L) <= model.Ca_j[j, t]

model.proc_capacity_constraint = pyo.Constraint(model.J, model.T, rule=proc_capacity_rule)

# Recovery method distribution
def recovery_rule(model, j, l, t):
    return model.y[j, l, t] == sum(model.y_r[j, l, t, r] for r in model.R)

model.recovery_constraint = pyo.Constraint(model.J, model.L, model.T, rule=recovery_rule)

# Quality threshold constraint (with slack)
def quality_rule(model, j, l, t, r):
    return model.y_r[j, l, t, r] <= model.QT[r] * model.SH[j, l, t] * model.y[j, l, t] + model.slack_quality[j, l, t, r]

model.quality_constraint = pyo.Constraint(model.J, model.L, model.T, model.R, rule=quality_rule)

# Material recovery calculation
def material_recovery_rule(model, m, j, l, t):
    return model.Rm[m, j, l, t] == sum(model.RE[m, r] * model.MC[m, l] * model.y_r[j, l, t, r] for r in model.R)

model.material_recovery_constraint = pyo.Constraint(model.M, model.J, model.L, model.T, rule=material_recovery_rule)

# Emissions limit
def emissions_rule(model):
    emissions = sum(model.e[j, l, t] * model.y[j, l, t] for j in model.J for l in model.L for t in model.T) + \
                sum(model.f[i, j, p] * sum(model.z[i, j, l, p, t] for l in model.L for t in model.T) 
                    for i in model.I for j in model.J for p in model.P)
    return emissions <= model.Me

model.emissions_constraint = pyo.Constraint(rule=emissions_rule)

# Inventory balance - Fixed to handle first time period
def inventory_rule(model, j, l, t):
    if t == 0:
        # Initial inventory is 0
        return model.In[j, l, t] == model.y[j, l, t] - model.Sh[j, l, t]
    else:
        return model.In[j, l, t] == model.In[j, l, t-1] + model.y[j, l, t] - model.Sh[j, l, t]

model.inventory_constraint = pyo.Constraint(model.J, model.L, model.T, rule=inventory_rule)

# Storage capacity
def storage_rule(model, j, t):
    return sum(model.In[j, l, t] for l in model.L) <= model.SC[j]

model.storage_constraint = pyo.Constraint(model.J, model.T, rule=storage_rule)

# Facility setup linkage
def setup_rule(model, j, l, t):
    big_M = 10000  # A sufficiently large number
    return model.y[j, l, t] <= big_M * model.b[j, l]

model.setup_constraint = pyo.Constraint(model.J, model.L, model.T, rule=setup_rule)

# Budget constraint
def budget_rule(model):
    setup_costs = sum(model.s[j, l] * model.b[j, l] for j in model.J for l in model.L)
    operating_costs = sum(model.Oc[j, t] for j in model.J for t in model.T)
    return setup_costs + operating_costs <= model.TB

model.budget_constraint = pyo.Constraint(rule=budget_rule)

# Solve the model
solver = pyo.SolverFactory('glpk')  # Or use another solver like 'cplex', 'gurobi', etc.

print("Solving deterministic model...")
results = solver.solve(model, tee=True)
print("Status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)

# Print key results
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("\nKey Results:")
    print("Objective Value:", pyo.value(model.objective))
    
    # Check if slack variables were used (which indicates infeasibility in original model)
    slack_usage = sum(pyo.value(model.slack_demand[l, t]) for l in model.L for t in model.T) + \
                 pyo.value(model.slack_process) + \
                 sum(pyo.value(model.slack_quality[j, l, t, r]) for j in model.J for l in model.L for t in model.T for r in model.R)
    
    print("\nSlack Usage:", slack_usage)
    if slack_usage > 0.001:  # Using a small threshold due to potential floating-point errors
        print("Warning: Slack variables are being used, indicating that the original model would be infeasible.")
        print("Slack for demand:", sum(pyo.value(model.slack_demand[l, t]) for l in model.L for t in model.T))
        print("Slack for processing:", pyo.value(model.slack_process))
        print("Slack for quality:", sum(pyo.value(model.slack_quality[j, l, t, r]) for j in model.J for l in model.L for t in model.T for r in model.R))
    
    # Print facility setup decisions
    print("\nFacility Setup Decisions:")
    for j in model.J:
        for l in model.L:
            if pyo.value(model.b[j, l]) > 0.5:  # Using > 0.5 to handle binary variables with small rounding errors
                print(f"Facility {j} is set up for battery type {l}")
    
    # Print collection and processing quantities
    print("\nTotal Collection by Period:")
    for t in model.T:
        print(f"Period {t}: {sum(pyo.value(model.x[i, l, t]) for i in model.I for l in model.L):.2f} units")
    
    print("\nTotal Processing by Period:")
    for t in model.T:
        print(f"Period {t}: {sum(pyo.value(model.y[j, l, t]) for j in model.J for l in model.L):.2f} units")
    
    # Print recovery quantities by method
    print("\nRecovery by Method:")
    for r in model.R:
        total = sum(pyo.value(model.y_r[j, l, t, r]) for j in model.J for l in model.L for t in model.T)
        print(f"Recovery method {r}: {total:.2f} units")
    
    # Print material recovery
    print("\nTotal Material Recovery:")
    for m in model.M:
        total = sum(pyo.value(model.Rm[m, j, l, t]) for j in model.J for l in model.L for t in model.T)
        print(f"Material {m}: {total:.2f} units")
else:
    print("No optimal solution found.")