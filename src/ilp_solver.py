import gurobipy as gp
from gurobipy import GRB


def pair(i, j):
    return (min(i, j), max(i, j))


def edge_exists(S, i, j):
    return S[i, j] != 0


def solve_ilp(
    S,
    time_limit=None,
    verbose=True,
    relax=False,
    add_four_cycles=True
):
    """
    Sparse ILP/LP relaxation for incomplete correlation clustering.

    Sparse means:
    - x variables only exist for observed edges, so S[i,j] != 0
    - triangle constraints are only added if all 3 triangle edges exist
    - bad 4-cycle constraints can be added separately
    """

    n = S.shape[0]

    if relax:
        model = gp.Model("ilp_relaxation")
    else:
        model = gp.Model("ilp")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    x = {}

    # Create variables only for existing edges
    for i in range(n):
        for j in range(i + 1, n):
            if edge_exists(S, i, j):
                if relax:
                    x[i, j] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                        name=f"x_{i}_{j}"
                    )
                else:
                    x[i, j] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{i}_{j}"
                    )

    model.update()

    # Objective: minimize disagreements only on existing edges
    objective_terms = []

    for (i, j), var in x.items():
        sign = S[i, j]

        if sign == 1:
            # positive edge pays if cut
            objective_terms.append(var)

        elif sign == -1:
            # negative edge pays if not cut
            objective_terms.append(1 - var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

    # Triangle inequalities only when all 3 edges exist
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):

                e_ij = pair(i, j)
                e_ik = pair(i, k)
                e_jk = pair(j, k)

                if e_ij in x and e_ik in x and e_jk in x:
                    x_ij = x[e_ij]
                    x_ik = x[e_ik]
                    x_jk = x[e_jk]

                    model.addConstr(x_ij <= x_ik + x_jk)
                    model.addConstr(x_ik <= x_ij + x_jk)
                    model.addConstr(x_jk <= x_ij + x_ik)

    bad_cycles = []

    if add_four_cycles:
        bad_cycles = add_sparse_four_cycle_constraints(
            model,
            x,
            S,
            n,
            print_cycles=False
        )

    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Gurobi did not solve successfully. Status: {model.Status}")

    objective_value = model.ObjVal

    if relax:
        x_values = {
            edge: var.X
            for edge, var in x.items()
        }
    else:
        x_values = {
            edge: int(round(var.X))
            for edge, var in x.items()
        }

    return objective_value, x_values, bad_cycles


def add_sparse_four_cycle_constraints(model, x, S, n, print_cycles=False):
    """
    Add bad 4-cycle constraints for sparse incomplete graphs.

    A bad 4-cycle means:
    - all 4 cycle edges exist
    - exactly 1 of the 4 cycle edges is negative
    - both diagonals are missing
    - each set of 4 vertices is counted once
    """

    bad_cycles = []
    seen_vertex_sets = set()

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):

                    cycles = [
                        [a, b, c, d],
                        [a, b, d, c],
                        [a, c, b, d],
                    ]

                    for cycle in cycles:
                        v1, v2, v3, v4 = cycle

                        e12 = pair(v1, v2)
                        e23 = pair(v2, v3)
                        e34 = pair(v3, v4)
                        e41 = pair(v4, v1)

                        cycle_edges = [e12, e23, e34, e41]

                        # All 4 cycle edges must exist as x variables
                        if not all(edge in x for edge in cycle_edges):
                            continue

                        diagonal_1 = pair(v1, v3)
                        diagonal_2 = pair(v2, v4)

                        # Both diagonals must be missing
                        diagonal_1_missing = diagonal_1 not in x
                        diagonal_2_missing = diagonal_2 not in x

                        if not (diagonal_1_missing and diagonal_2_missing):
                            continue

                        signs = [S[i, j] for i, j in cycle_edges]

                        # Bad cycle: exactly 1 negative edge
                        num_negative_edges = sum(1 for sign in signs if sign == -1)

                        if num_negative_edges != 1:
                            continue

                        vertex_set = tuple(sorted(cycle))

                        if vertex_set in seen_vertex_sets:
                            continue

                        seen_vertex_sets.add(vertex_set)

                        bad_cycles.append(
                            (cycle, cycle_edges, signs, diagonal_1, diagonal_2)
                        )

                        x_12 = x[e12]
                        x_23 = x[e23]
                        x_34 = x[e34]
                        x_41 = x[e41]

                        model.addConstr(
                            x_12 <= x_23 + x_34 + x_41,
                            name=f"sparse_bad_four_cycle_{len(bad_cycles)}_1"
                        )

                        model.addConstr(
                            x_23 <= x_12 + x_34 + x_41,
                            name=f"sparse_bad_four_cycle_{len(bad_cycles)}_2"
                        )

                        model.addConstr(
                            x_34 <= x_12 + x_23 + x_41,
                            name=f"sparse_bad_four_cycle_{len(bad_cycles)}_3"
                        )

                        model.addConstr(
                            x_41 <= x_12 + x_23 + x_34,
                            name=f"sparse_bad_four_cycle_{len(bad_cycles)}_4"
                        )

    if print_cycles:
        print("Sparse bad 4-cycles found:")

        for number, (cycle, cycle_edges, signs, diagonal_1, diagonal_2) in enumerate(bad_cycles, start=1):
            print(f"{number}. vertices: {cycle}")
            print(f"   cycle edges: {cycle_edges}")
            print(f"   signs: {signs}")
            print(f"   missing diagonals: {diagonal_1}, {diagonal_2}")
            print()

        print(f"Total sparse bad 4-cycles found: {len(bad_cycles)}")

    return bad_cycles

def find_ilp_clusters(x_values, n):
    """
    Given sparse ILP x_values, returns clusters.

    x[i,j] = 0 means i and j are in the same cluster.
    Missing edges are not in x_values and are therefore not used directly.
    """
    clusters = [{i} for i in range(n)]

    for (i, j), val in x_values.items():
        if val == 0:
            ci = cj = None

            for cluster in clusters:
                if i in cluster:
                    ci = cluster
                if j in cluster:
                    cj = cluster

            if ci is not cj:
                merged = ci.union(cj)
                clusters.remove(ci)
                clusters.remove(cj)
                clusters.append(merged)

    return clusters
