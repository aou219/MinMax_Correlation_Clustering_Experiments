def is_bad_triangle(S, i, j, k):
    return(
        (S[i, j] == 1 and S[i, k] == 1 and S[j, k] == -1)
        or (S[i,j] == 1 and S[i,k]==-1 and S[j,k]== 1)
        or (S[i,j] == -1 and S[i,k]==1 and S[j,k]==1)
    )
"""
Right now, you search for all bad triangles, including those that share edges with other bad triangles.
This does not make a good lower bound. Try searching for all edge disjoint bad triangles.
"""
def find_bad_triangles(S):
    n = S.shape[0]
    bad_triangle_list = []
    # To keep an order, we force the first node to be the smallest, then the second node, then the last.
    # This way 012 will get iterated and 021 won't
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1, n):
                if(is_bad_triangle(S, i, j, k)):
                    bad_triangle_list.append((i,j,k))
    return bad_triangle_list

def count_bad_triangles(bad_triangle_list):
    return len(bad_triangle_list)

def bad_triangles_containing_edge(bad_triangle_list,  n1, n2):
    triangles_with_edge = []
    edge = { n1,n2}
    for triangle in bad_triangle_list:
        i,j,k = triangle #unpacking
        triangle_edges = [
            {i, j},
            {i, k},
            {k, j}
        ]
        if edge in triangle_edges:
            triangles_with_edge.append((triangle))
    return triangles_with_edge