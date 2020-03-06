from pqdict import PQDict


def dijkstra(G, start, end=None):
    '''
    dijkstra's algorithm determines the length from `start` to every other
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]`
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1},
                 'b': {'c': 2, 'b': 5},
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a')

    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}

    '''
    inf = float('inf')
    D = {start: 0}  # mapping of nodes to their dist from start
    Q = PQDict(D)  # priority queue for tracking min shortest path
    P = {}  # mapping of nodes to their direct predecessors
    U = set(G.keys())  # unexplored nodes

    while U:  # nodes yet to explore
        (v, d) = Q.popitem()  # node w/ min dist d on frontier
        D[v] = d  # est dijkstra greedy score
        U.remove(v)  # remove from unexplored
        if v == end: break

        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors
        for w in G[v]:  # successors to v
            if w in U:  # then w is a frontier node
                d = D[v] + G[v][w]  # dgs: dist of start -> v -> w
                if d < Q.get(w, inf):
                    Q[w] = d  # set/update dgs
                    P[w] = v  # set/update predecessor

    return D, P


def shortest_path(G, start, end):
    dist, pred = dijkstra(G, start, end)
    v = end
    path = [v]
    while v != start:
        v = pred[v]
        path.append(v)
    path.reverse()
    return path


if __name__ == '__main__':
    graph = {'a': {'b': 1},
             'b': {'c': 2, 'b': 5},
             'c': {'d': 1},
             'd': {}}

    # get shortest path distances to each node in `graph` from `a`
    dist, pred = dijkstra(graph, 'a')
    assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}  # min dist from `a`
    assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}  # direct predecessors
    assert shortest_path(graph, 'a', 'd') == list('abcd')

    graph = {'a': {'b': 14, 'c': 9, 'd': 7},
             'b': {'a': 14, 'c': 2, 'e': 9},
             'c': {'a': 9, 'b': 2, 'd': 10, 'f': 11},
             'd': {'a': 7, 'c': 10, 'f': 15},
             'e': {'b': 9, 'f': 6},
             'f': {'c': 11, 'd': 15, 'e': 6}}

    dist, pred = dijkstra(graph, start='a')
    expected = {'a': 0, 'c': 9, 'b': 11, 'e': 20, 'd': 7, 'f': 20}
    assert dist == expected
