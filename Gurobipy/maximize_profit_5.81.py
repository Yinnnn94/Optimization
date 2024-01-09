import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("maximize_profit")

x = m.addVar(name = "x")

m.setObjective(3000 + 500 * x - 100 * x * x, GRB.MAXIMIZE)
m.addConstr(x >= 0, "c0")
m.addConstr(x <= 10, "c1")

m.optimize()
price = 1.5 - x.x * 0.1
number = 5000 + 1000 * x.x
print(f'Objecttive: {m.objVal}')
print(f'x: {x.x}')
print(f'Number: {number}')
print(f'Price: {price}')
