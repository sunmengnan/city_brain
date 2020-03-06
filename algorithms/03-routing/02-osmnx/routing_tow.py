import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KDTree
import folium
import matplotlib.pyplot as plt

sorth_street = (30.207238, 120.204248)
G = ox.graph_from_point(sorth_street, distance=500)
ox.plot_graph(G, fig_height=10, fig_width=10, edge_color='green')

route = nx.shortest_path(G, np.random.choice(G.nodes),
                         np.random.choice(G.nodes))
ox.plot_graph_route(G, route, fig_height=10, fig_width=10)

nodes, _ = ox.graph_to_gdfs(G)
print(nodes)

node_a = (30.205538, 120.199288)
node_b = (30.212378, 120.211898)
node_c = (30.20928, 120.21248)
node_d = (30.20158, 120.20588)

tree = KDTree(nodes[['y', 'x']], metric='euclidean')
a_idx = tree.query([node_a], k=1, return_distance=False)[0]
b_idx = tree.query([node_b], k=1, return_distance=False)[0]
c_idx = tree.query([node_c], k=1, return_distance=False)[0]
closest_node_to_a = nodes.iloc[a_idx].index.values[0]
closest_node_to_b = nodes.iloc[b_idx].index.values[0]
closest_node_to_c = nodes.iloc[c_idx].index.values[0]

fig, ax = ox.plot_graph(G, fig_height=10, fig_width=10,
                        show=False, close=False,
                        edge_color='black')

ax.scatter(G.node[closest_node_to_a]['x'],
           G.node[closest_node_to_a]['y'],
           c='green', s=100)
ax.scatter(G.node[closest_node_to_b]['x'],
           G.node[closest_node_to_b]['y'],
           c='green', s=100)
ax.scatter(G.node[closest_node_to_c]['x'],
           G.node[closest_node_to_c]['y'],
           c='green', s=100)
plt.show()

route = nx.shortest_path(G, closest_node_to_a,
                         closest_node_to_c)
fig, ax = ox.plot_graph_route(G, route, fig_height=10,
                              fig_width=10,
                              show=False, close=False,
                              edge_color='black',
                              orig_dest_node_color='green',
                              route_color='green')
plt.show()
