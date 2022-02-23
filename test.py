import itertools

layer_num = 3
layer_options = [10*(i+1) for i in range(8)]
print(f"layer pos: {layer_options}")
activations = ["elu", "tanh", "sigmoid"]
layer_possibilities = [layer_options] * layer_num
activations_posibilities = activations* layer_num
for i in range(layer_num+1):
    layer_possibilities.append(activations)
layer_node_permutations = list(itertools.product(*layer_possibilities))
#print(layer_node_permutations)
print(len(layer_node_permutations))

one_layer = [layer_options, activations]
one_layer_permutations = list(itertools.product(*one_layer))
print(one_layer_permutations)
print(len(one_layer_permutations))
