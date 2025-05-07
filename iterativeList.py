class Node:
    def __init__(self, op_type, inputs=None, name=None):
        """
        op_type: 'var', '+', '*'
        inputs: list of node indices this operation applies to (for 'var', None)
        name: variable name (for vars only)
        """
        self.op_type = op_type
        self.inputs = inputs or []
        self.name = name  # Used only if op_type == 'var'
        self.output = None  # can store computed result here

    def __repr__(self):
        if self.op_type == 'var':
            return f"{self.name}"
        else:
            return f"{self.op_type} {self.inputs[0]} {self.inputs[1]}"


class ComputationList:
    def __init__(self, enable_cache=True):
        self.nodes = []
        self.enable_cache = enable_cache

    def add_variable(self, name):
        node = Node('var', name=name)
        self.nodes.append(node)
        return len(self.nodes) - 1  # Return index

    def add_operation(self, op_type, left_idx, right_idx):
        assert op_type in ('+', '*'), "Only '+' and '*' are supported"
        node = Node(op_type, inputs=[left_idx, right_idx])
        self.nodes.append(node)
        return len(self.nodes) - 1

    def clear_cache(self):
        for node in self.nodes:
            node.output = None

    def __str__(self):
        output = []
        for idx, node in enumerate(self.nodes, start=1):
            if node.op_type == 'var':
                output.append(f"{idx} {node.name}")
            else:
                output.append(f"{idx} {node.op_type} {node.inputs[0]+1} {node.inputs[1]+1}")
        return "\n".join(output)


# Example Usage

# (x + y)^2
example1 = ComputationList()
x = example1.add_variable("x")
y = example1.add_variable("y")
add1 = example1.add_operation("+", x, y)
result1 = example1.add_operation("*", add1, add1)
print("Example 1:")
print(example1)

print("\n")

# x_1*(x_2+x_3)+x_4^2*(x_1+x_3)
example2 = ComputationList()
x_1 = example2.add_variable("x_1")
x_2 = example2.add_variable("x_2")
x_3 = example2.add_variable("x_3")
x_4 = example2.add_variable("x_4")
x_2_add_x_3 = example2.add_operation("+", x_2, x_3)
x_1_mul_sum = example2.add_operation("*", x_1, x_2_add_x_3)
x_4_squared = example2.add_operation("*", x_4, x_4)
x_1_add_x_3 = example2.add_operation("+", x_1, x_3)
squared_mul_sum = example2.add_operation("*", x_4_squared, x_1_add_x_3)
result2 = example2.add_operation("+", x_1_mul_sum, squared_mul_sum)
print("Example 2:")
print(example2)

