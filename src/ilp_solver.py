import gurobipy as gp
from gurobipy import GRB
import networkx as nx

# Actually not needed because the way I iterate already forces it to be on order.
def pair(i, j):
    return (min(i, j), max(i, j))


def solve_correlation_clustering_ilp(S, time_limit=None, verbose=True):
    """
    Solve the exact correlation clustering ILP.

    x[i,j] = 0 means i and j are in the same cluster
    x[i,j] = 1 means i and j are in different clusters

    Cost:
    - positive edge cut: x[i,j]
    - negative edge inside cluster: 1 - x[i,j]
    """
    n = S.shape[0]

    model = gp.Model("correlation_clustering_ilp")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Create binary variables for every pair of vertices
    x = {}

    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    # Objective: minimize disagreements
    objective_terms = []

    for i in range(n):
        for j in range(i + 1, n):
            sign = S[i, j]

            if sign == 1:
                objective_terms.append(x[i, j])

            elif sign == -1:
                objective_terms.append(1 - x[i, j])

            elif sign == 0:
                continue

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

    # Triangle inequalities
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x_ij = x[pair(i, j)]
                x_ik = x[pair(i, k)]
                x_jk = x[pair(j, k)]

                model.addConstr(x_ij <= x_ik + x_jk)
                model.addConstr(x_ik <= x_ij + x_jk)
                model.addConstr(x_jk <= x_ij + x_ik)

    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Gurobi did not solve successfully. Status: {model.Status}")

    objective_value = model.ObjVal

    x_values = {
        edge: int(round(var.X))
        for edge, var in x.items()
    }

    return objective_value, x_values