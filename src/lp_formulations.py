import gurobipy as gp
from gurobipy import GRB
from bad_triangles import find_bad_triangles


def solve_primal(S, bad_triangles, time_limit=None, verbose=True):
    n = S.shape[0]

    model = gp.Model("lp_bad_triangle")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = {}

    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x_{i}_{j}")

    model.update()

    objective_terms = [x[i,j] for i in range(n) for j in range(i+1, n)]
    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

    #triangle constraints
    for triangle in bad_triangles:
        i, j, k = triangle
        model.addConstr(x[i,j] + x[j,k] + x[i,k] >= 1,
                        name=f"triangle_{i}_{j}_{k}")

    model.optimize()

    # Extract solution
    x_values = {(i,j): var.X for (i,j), var in x.items()}

    return model.ObjVal, x_values

def solve_dual(S, bad_triangles, time_limit=None, verbose=True):
    n = S.shape[0]

    model = gp.Model("lp_dual")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Create a variable for each bad triangle
    y = {}
    # give an index to each bad triangle, attach to that index some Gurobi variable that needs to be optimized later
    for t_index, triangle in enumerate(bad_triangles):
        y[t_index] = model.addVar(lb=0.0, ub=1.0, name=f"y_{triangle}")

    model.update()

    # Constraints: sum of y_t over triangles containing each edge <= 1
    # make a dictionary: edge->triangles(according to the t_index)
    # for every triangle:
    #  for every edge in the triangle
    #   if it is not yet in the dictionary, add it and add the corresponding triangle
    edge_to_triangles = {}
    for t_index, (i, j, k) in enumerate(bad_triangles):
        for u, v in [(i, j), (i, k), (j, k)]:
            edge = tuple(sorted((u, v)))
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(t_index)

# Sums up the dual variables y[t_index] for all triangles that contain the current edge.
# This is the total “weight” assigned to triangles that share this edge in the dual LP and should be ≤1
    for edge, triangles in edge_to_triangles.items():
        model.addConstr(gp.quicksum(y[t_index] for t_index in triangles) <= 1)

    # Objective: maximize sum of y_t
    model.setObjective(gp.quicksum(y.values()), GRB.MAXIMIZE)

    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Gurobi did not solve successfully. Status: {model.Status}")

    objective_value = model.ObjVal
    y_values = {t_index: var.X for t_index, var in y.items()}

    return objective_value, y_values