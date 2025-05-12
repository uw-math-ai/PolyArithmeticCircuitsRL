class GraphNode:
    def __init__(self, node_id, op_type, name=None):
        self.id = node_id
        self.op_type = op_type  # 'var', '+', '*'
        self.name = name        # only used if op_type == 'var'
        self.inputs = []        # list of node ids
        self.outputs = []       # list of node ids
        self.value = None       # cached value if needed

    def __repr__(self):
        if self.op_type == 'var':
            return f"Node({self.id}): var '{self.name}'"
        else:
            return f"Node({self.id}): {self.op_type}({', '.join(map(str, self.inputs))})"


class ComputationGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.next_id = 0

    def add_variable(self, name):
        node = GraphNode(self.next_id, 'var', name)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.id

    def add_operation(self, op_type, left_id, right_id):
        assert op_type in ('+', '*')
        node = GraphNode(self.next_id, op_type)
        node.inputs = [left_id, right_id]
        self.nodes[self.next_id] = node

        # Update connections
        self.nodes[left_id].outputs.append(node.id)
        self.nodes[right_id].outputs.append(node.id)
        self.edges.append((left_id, node.id))
        self.edges.append((right_id, node.id))

        self.next_id += 1
        return node.id

    def __repr__(self):
        result = []
        for node_id in sorted(self.nodes):
            result.append(str(self.nodes[node_id]))
        return "\n".join(result)

    def visualize_edges(self):
        return "\n".join(f"{src} -> {dst}" for src, dst in self.edges)


g = ComputationGraph()
x = g.add_variable("x")
y = g.add_variable("y")
z = g.add_operation('*', x, x)
w = g.add_operation('+', y, y)
p = g.add_operation('*', z, w)

print(g)
print("--- Edges ---")
print(g.visualize_edges())