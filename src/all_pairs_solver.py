import gurobipy as gp
from gurobipy import GRB


def pair(i, j):
    return min(i, j), max(i, j)


def solve_all_pairs(
    S,
    time_limit=None,
    verbose=True,
    relax=False,
    return_x_values=True,
):
    """Solve the full metric ILP or its LP relaxation."""
    n = S.shape[0]
    model_name = "all_pairs_lp" if relax else "all_pairs_ilp"
    model = gp.Model(model_name)

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    variable_type = GRB.CONTINUOUS if relax else GRB.BINARY
    x = {
        (i, j): model.addVar(
            vtype=variable_type,
            lb=0,
            ub=1,
            name=f"x_{i}_{j}",
        )
        for i in range(n)
        for j in range(i + 1, n)
    }

    objective_terms = []
    for (i, j), variable in x.items():
        if S[i, j] == 1:
            objective_terms.append(variable)
        elif S[i, j] == -1:
            objective_terms.append(1 - variable)
    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

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

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(
            f"Gurobi did not solve successfully. Status: {model.Status}"
        )
    if model.SolCount == 0:
        raise RuntimeError("Gurobi stopped without a feasible solution.")

    x_values = None
    if return_x_values:
        if relax:
            x_values = {edge: variable.X for edge, variable in x.items()}
        else:
            x_values = {
                edge: int(round(variable.X)) for edge, variable in x.items()
            }

    solve_info = {
        "status": int(model.Status),
        "is_optimal": model.Status == GRB.OPTIMAL,
        "runtime_seconds": float(model.Runtime),
        "mip_gap": float(model.MIPGap) if not relax else None,
        "num_variables": len(x),
        "num_triangle_constraints": 3 * (n * (n - 1) * (n - 2) // 6),
    }
    objective_value = float(model.ObjVal)
    model.dispose()
    return objective_value, x_values, solve_info
