# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:55:39 2023

@author: dclisu
"""

import networkx as nx
import matplotlib.pyplot as plt

# Create four example networks

n = 15
avg_degree = 4


G1 = nx.complete_graph(n)
G2 = nx.expected_degree_graph([avg_degree for i in range(n)],selfloops = False)
G3 = nx.connected_caveman_graph(3, 5)
G4 = nx.windmill_graph(2, 8)

# Create a 2x2 subplot and plot each network in a separate axis
fig, axs = plt.subplots(1, 4, figsize=(21, 4))
nx.draw(G1, ax=axs[0],node_color = 'gray')
axs[0].set_title('Complete',fontsize = 30)
nx.draw(G2, ax=axs[1],node_color = 'gray')
axs[1].set_title('Uniform random',fontsize = 30)
nx.draw(G3, ax=axs[2],node_color = 'gray')
axs[2].set_title('Caveman',fontsize = 30)
nx.draw(G4, ax=axs[3],node_color = 'gray')
axs[3].set_title('Windmill',fontsize = 30)

# Add a title to the plot
#fig.suptitle("Four example networks")

# Display the plot
plt.savefig('topology_examples.eps', format='eps')
plt.show()