# This page explains how the compact graph encoding works

## Assumptions

1) The optimal circuit requires at most N nodes (excluding input nodes) to represent the polynomial. This enables the compact representation 
2) The Polynomial lives in Z_P[x_1, x_2, ..., x_D]. This allows us to handle constants and predetermine the number of input nodes

## RL Framework

The state consists of the polynomial and current (potentially incomplete) arithmetic circuit that has been built
The action consists of two previous nodes in the circuit along with an operator
The reward isn't too important but has something to do with building the correct circuit

## Idea

Conceptually, imagine there are n nodes that are all disconnected. We also further have D input nodes for each variable and P-1 input nodes for all constants. Let the set of "known" nodes begin as these input nodes. This "known" set will essentially consist of all the nodes for which we know what polynomial the node represents. We enforce that an action will operate on nodes in the "known" set and connect to the next available node that is not in the "known" set. In order for this operation to occur, the action must only describe the id of two nodes (from the known set) along with the operation type. Given this, it is then possible to determine what polynomial the new node will represent. We also need to keep track of what node is the "next" node. We may then state that the polynomial that is represented by an incomplete circuit is the polynomial being represented by the most recently operated on node. Note that this means our conceptual graph is NOT a simple graph (there may be two connections from the same node to a node later on if it is reused!). It is a directed graph, and specifically, directed edges can only connect a node with smaller id to a node with greater id by construction. 

Making this idea concrete, to encode the circuit we only need to know which directed edges have been connected and what the id of the current node is. Then, we may one-hot encode all of these features for stability in training. Our encoding becomes a vector in Z_2^[2 * N + N * [ 2 * I + N - 1] + N], where I = D + P - 1 is the number of input nodes. The first 2N nodes one hot encode the operation type of the N nodes. The next 2 * (I + ... + I + N - 1) nodes one hot encode the connections to the N nodes. For example, the first I nodes and I nodes following that one hot encode which two nodes connect to the first node. The next I + 1 nodes and I + 1 nodes following that one hot encode which two noes connect to the second node, etc. The final N nodes one hot encoder the id of the node that currently represents the polynomial.

## Notes

There are concerns about how the encoding dimension is O(N^2). To address this one can dismiss the one hot encoding of edges in favor of a smaller encoding.  