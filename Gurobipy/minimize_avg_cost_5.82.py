import gurobipy as gp
from gurobipy import GRB


m = gp.Model("minimize_avg_cost")
q = m.addVar(name = "q")
m3 = m.addVar(name = "m3")
m.setObjective(0.0001 * q * q - 0.08 * q + 65 + m3, GRB.MINIMIZE)
m.addConstr(q >= 0, 'c0')
m.addConstr(5000 == m3 * q, 'c1')
m.optimize()

print(f"Optimal solution: q = {m.ObjVal}")
print(f"Optimal solution: q = {q}")