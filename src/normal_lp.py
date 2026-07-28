"""LP solver for normal correlation clustering."""

import gurobipy as gp
from gurobipy import GRB


def solve_normal_lp(
    S,
    time_limit=None,
    verbose=True,
):
    """Solve the LP relaxation for normal correlation clustering."""
    n = S.shape[0]
    model = gp.Model("normal_cc_lp")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    x = {
        (i, j): model.addVar(
            vtype=GRB.CONTINUOUS,
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

    model.setObjective(
        gp.quicksum(objective_terms),
        GRB.MINIMIZE,
    )

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                model.addConstr(
                    x[i, j] <= x[i, k] + x[j, k]
                )
                model.addConstr(
                    x[i, k] <= x[i, j] + x[j, k]
                )
                model.addConstr(
                    x[j, k] <= x[i, j] + x[i, k]
                )

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(
            f"Gurobi did not solve successfully. "
            f"Status: {model.Status}"
        )
    if model.SolCount == 0:
        raise RuntimeError(
            "Gurobi stopped without a feasible solution."
        )

    cost = float(model.ObjVal)
    info = {
        "status": int(model.Status),
        "is_optimal": model.Status == GRB.OPTIMAL,
        "runtime_seconds": float(model.Runtime),
        "num_variables": len(x),
        "num_triangle_constraints": (
            3 * (n * (n - 1) * (n - 2) // 6)
        ),
    }

    model.dispose()
    return cost, info
