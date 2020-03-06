import time

from shapely.geometry import LineString, Point
from osmnx import log, is_simplified, lg, build_path


def is_endpoint(G, node, strict=True, other_endpoints=None):
    """
    Return True if the node is a "real" endpoint of an edge in the network, \
    otherwise False. OSM data includes lots of nodes that exist only as points \
    to help streets bend around curves. An end point is a node that either: \
    1) is its own neighbor, ie, it self-loops. \
    2) or, has no incoming edges or no outgoing edges, ie, all its incident \
        edges point inward or all its incident edges point outward. \
    3) or, it does not have exactly two neighbors and degree of 2 or 4. \
    4) or, if strict mode is false, if its edges have different OSM IDs. \

    Parameters
    ----------
    G : networkx multidigraph

    node : int
        the node to examine
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules \
        but have edges with different OSM IDs

    Returns
    -------
    bool

    """
    if other_endpoints is None:
        other_endpoints = []

    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # y
    # if a node is osm u,v point,it's endpoint,use this to cut edge
    # if node in G.nodes():
    #     return True
    # pre calc end_points to cut.
    if node in other_endpoints:
        print('123')
        return True

    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops. this is
        # always an endpoint.
        return True

    # if node has no incoming edges or no outgoing edges, it must be an endpoint
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        return True

    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    elif not strict:
        # non-strict mode
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]['osmid'])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]['osmid'])

        # if there is more than 1 OSM ID in the list of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    else:
        # if none of the preceding rules returned true, then it is not an endpoint
        return False


def get_paths_to_simplify(G, strict=True, other_endpoints=None):
    """
    Create a list of all the paths to be simplified between endpoint nodes.

    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint. If your street network is in a rural area with many
    interstitial nodes between true edge endpoints, you may want to increase
    your system's recursion limit to avoid recursion errors.

    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs

    Returns
    -------
    paths_to_simplify : list
    """
    if other_endpoints is None:
        other_endpoints = []

    # first identify all the nodes that are endpoints
    start_time = time.time()
    endpoints = set(
        [node for node in G.nodes() if is_endpoint(G, node, strict=strict, other_endpoints=other_endpoints)])
    log('Identified {:,} edge endpoints in {:,.2f} seconds'.format(len(endpoints), time.time() - start_time))

    start_time = time.time()
    paths_to_simplify = []

    # for each endpoint node, look at each of its successor nodes
    for node in endpoints:
        for successor in G.successors(node):
            if successor not in endpoints:
                # if the successor is not an endpoint, build a path from the
                # endpoint node to the next endpoint node
                try:
                    path = build_path(G, successor, endpoints, path=[node, successor])
                    paths_to_simplify.append(path)
                except RuntimeError:
                    log('Recursion error: exceeded max depth, moving on to next endpoint successor', level=lg.WARNING)
                    # recursion errors occur if some connected component is a
                    # self-contained ring in which all nodes are not end points.
                    # could also occur in extremely long street segments (eg, in
                    # rural areas) with too many nodes between true endpoints.
                    # handle it by just ignoring that component and letting its
                    # topology remain intact (this should be a rare occurrence)
                    # RuntimeError is what Python <3.5 will throw, Py3.5+ throws
                    # RecursionError but it is a subtype of RuntimeError so it
                    # still gets handled

    log('Constructed all paths to simplify in {:,.2f} seconds'.format(time.time() - start_time))
    return paths_to_simplify


def simplify_graph_ext1(G, strict=False, other_endpoints=None):
    """
    y:
    logicaly.strict=True not cause any different result

    Simplify a graph's topology by removing all nodes that are not intersections
    or dead-ends.

    Create an edge directly between the end points that encapsulate them,
    but retain the geometry of the original edges, saved as attribute in new
    edge.

    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs

    Returns
    -------
    networkx multidigraph
    """
    if other_endpoints is None:
        other_endpoints = []

    if is_simplified(G):
        raise Exception('This graph has already been simplified, cannot simplify it again.')

    log('Begin topologically simplifying the graph...')
    G = G.copy()
    initial_node_count = len(list(G.nodes()))
    initial_edge_count = len(list(G.edges()))
    all_nodes_to_remove = []
    all_edges_to_add = []

    # construct a list of all the paths that need to be simplified
    paths = get_paths_to_simplify(G, strict=strict, other_endpoints=other_endpoints)

    start_time = time.time()
    for path in paths:

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        edge_attributes = {}
        for u, v in zip(path[:-1], path[1:]):

            # there shouldn't be multiple edges between interstitial nodes
            if not G.number_of_edges(u, v) == 1:
                log('Multiple edges between "{}" and "{}" found when simplifying'.format(u, v), level=lg.WARNING)

            # the only element in this list as long as above check is True
            # (MultiGraphs use keys (the 0 here), indexed with ints from 0 and
            # up)
            edge = G.edges[u, v, 0]
            for key in edge:
                if key in edge_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    edge_attributes[key].append(edge[key])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    edge_attributes[key] = [edge[key]]

        for key in edge_attributes:
            # don't touch the length attribute, we'll sum it at the end
            if len(set(edge_attributes[key])) == 1 and not key == 'length':
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th)
                edge_attributes[key] = edge_attributes[key][0]
            elif not key == 'length':
                # otherwise, if there are multiple values, keep one of each value
                edge_attributes[key] = list(set(edge_attributes[key]))

        # construct the geometry and sum the lengths of the segments
        edge_attributes['geometry'] = LineString([Point((G.nodes[node]['x'], G.nodes[node]['y'])) for node in path])
        edge_attributes['length'] = sum(edge_attributes['length'])
        # by y
        edge_attributes['_nodes'] = [node for node in path]

        # add the nodes and edges to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append({'origin': path[0],
                                 'destination': path[-1],
                                 'attr_dict': edge_attributes})

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge['origin'], edge['destination'], **edge['attr_dict'])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    G.graph['simplified'] = True

    msg = 'Simplified graph (from {:,} to {:,} nodes and from {:,} to {:,} edges) in {:,.2f} seconds'
    log(msg.format(initial_node_count, len(list(G.nodes())), initial_edge_count, len(list(G.edges())),
                   time.time() - start_time))
    return G
