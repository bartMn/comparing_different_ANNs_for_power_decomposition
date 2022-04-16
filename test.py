import itertools
import csv
"""
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

x= [1,2,3,4,5,6,7]
X = x.pop()
print(x)
print(X)
"""

"""
data = [("data0", 14), ("data1", 78), ("data3", 1)]
data = sorted(data, key=lambda ele: ele[1])

new = ("data4", 0)
data.append(new)
data = sorted(data, key=lambda ele: ele[1])
data.pop(-1)
print(data)


a0 = [("data0", 14), ("data1", 78), ("data3", 1)]
a1 = [("data4", 7), ("data5", 54), ("data6", 0)]

a = []
a +=  a0+ a1
a = sorted(a, key=lambda ele: ele[1])
print(a)

with open(f'test.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        data = ([1,2,3,4,5], [9, 888])
        data = data[0] + data[1]
        writer.writerow(data)
        
"""

x = [1, 2, 3]
x1 = x*3
print(x1)


number_steps_to_try = [i for i in range(1, 10)] + [10*j for j in range(1,19)]
hyperparameters = [96, 61, 126, "tanh", "elu", "relu", "relu"]
hyperparameters2 = [11, 76, 126, "sigmoid", "relu", "tanh", "sigmoid"]

LSTMs_to_try = []
LSTMs_to_try.append([hyperparameters, hyperparameters2])
LSTMs_to_try.append(number_steps_to_try)
"""    
LSTMs = list(itertools.product(*LSTMs_to_try))
print(len(LSTMs))
for L in LSTMs:
    print(locals())
temp = LSTMs[10]
print(temp[1])
print(list(temp[0]+[temp[1]]))
"""
Arr = [i for i in range(1, 10)] + [10*j for j in range(1,21)]

#print(Arr)

v1 = ["1,2","3,4"]
v2 = [5,6,7,8]
v3 = v1 + v2
print(v3[::-1])


