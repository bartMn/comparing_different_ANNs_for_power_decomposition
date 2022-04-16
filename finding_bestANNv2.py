#this profram tries to find the best FFN with specified hidden layers
#all tested FFNs' hyperparameters and their results are saved in csv file

from ann import ANN
import itertools
import csv
import time
import os

PATH = os.path.abspath(os.getcwd())

def write_ann_tested(ann_done, houses, train_method, ann_type, layers):
    
    with open(f'{houses}tested_{layers}_layer_{ann_type}_new_mes2_7days_20trained_{train_method}.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)      
        writer.writerow(ann_done)


def exploreANNs(layer_num, train_method, number_neurons_to_try, activations_to_try, ann_type, train, test, test_mode):
    
    test_sets = [train, test]
    
    print(number_neurons_to_try)
    layer_possibilities = [number_neurons_to_try] * layer_num
    
    for i in range(layer_num+1):
        layer_possibilities.append(activations_to_try)
    
    test_possibilities = [number_neurons_to_try]
    test_possibilities.append(activations_to_try)
    test_possibilities = list(itertools.product(*test_possibilities))
    
    
    top_anns = [[[1, "linear", "linear"], 9*(10**6)] for i in range(10)]
    
    #ann_type = "FFN"               #type of ANN
    layers = [8]                    #array of numbers of units in each layer 
    house_train = train             
    house_test = test
    house_test2 = test
    files_to_train = [f"total_P_new{house_train}", f"total_Q_new{house_train}", f"decomposition_new{house_train}"]  #files that are used to train ANN
    files_to_test =  [f"total_P_new{house_test}", f"total_Q_new{house_test}", f"decomposition_new{house_test}"]     #files that are used to make predictions
    file_to_write = "predictions"           #file to save predictions
    files_to_test2 = [f"total_P_new{house_test2}", f"total_Q_new{house_test2}", f"decomposition_new{house_test2}"]
    model_to_load = f"test_model{ann_type}" #file to load a model (only relevant if you want to load), make sure you load the same type (e.g. FFN)
    steps = 25                              #number of steps (only relevant for LSTM)
    
    model = ANN(ann_type = ann_type, layers = layers, files_to_train= files_to_train, files_to_test= files_to_test, file_to_write = file_to_write, steps = steps)
    model.add_testing_data(files_to_test2)
    
    start = time.time()
    best_model = []
    for i in range(layer_num):
        best_model.append(1)
    for i in range(layer_num+1):
        best_model.append("linear")
    best_mse = 999999999999999999999999999999
    rounds = 5
    
    all_tests = rounds*(layer_num*len(test_possibilities)+ len(activations_to_try))
    print(f"number of ANNs to test: {all_tests}")
    tests_done = 0
    
    
    for r in range(rounds):
        
        #####################
        test_model = list(best_model)
        for output_function in activations_to_try:
            test_model[-1] = output_function
            mse1, mse2, custom0, custom1, custom2, custom3, custom4, custom5 = model.create_test_ann(test_model, train_method)
            mse = [mse1, mse2]

            top_anns.append([test_model, mse[test_mode]])
            top_anns = sorted(top_anns, key=lambda ann: ann[1])
            top_anns.pop(-1)

            if mse[test_mode] < best_mse:
                best_model = list(test_model)
                best_mse = mse[test_mode]

            data_to_write = list(test_model)
            data_to_write.append(mse[test_mode])
            write_ann_tested(data_to_write, test_sets[test_mode], train_method, ann_type, layer_num)

            print(f"mse for {test_sets[test_mode]} houses = {mse[test_mode]}")

            tests_done += 1
            time_elapsed = time.time() - start
            fraction_done = tests_done/all_tests
            fraction_left = 1 - tests_done/all_tests 
            print(f"done {tests_done} out of {all_tests} tests")
            print(f"time elapsed: {int(time_elapsed/60)} minutes; time left: {int(time_elapsed*(fraction_left/fraction_done)/60)} minutes")  
        #####################
        
        for i in range(layer_num-1, -1, -1):
            
            test_model = list(best_model)
            for test_permutation in test_possibilities:
                test_model[i] = test_permutation[0]
                test_model[layer_num + i] = test_permutation[1]
                mse1, mse2, custom0, custom1, custom2, custom3, custom4, custom5 = model.create_test_ann(test_model, train_method)
                mse = [mse1, mse2]
                
                top_anns.append([test_model, mse[test_mode]])
                top_anns = sorted(top_anns, key=lambda ann: ann[1])
                top_anns.pop(-1)
                
                if mse[test_mode] < best_mse:
                    best_model = list(test_model)
                    best_mse = mse[test_mode]
                
                data_to_write = list(test_model)
                data_to_write.append(mse[test_mode])
                write_ann_tested(data_to_write, test_sets[test_mode], train_method, ann_type, layer_num)
                
                print(f"mse for {test_sets[test_mode]} houses = {mse[test_mode]}")
                
                tests_done += 1
                time_elapsed = time.time() - start
                fraction_done = tests_done/all_tests
                fraction_left = 1 - tests_done/all_tests 
                print(f"done {tests_done} out of {all_tests} tests")
                print(f"time elapsed: {int(time_elapsed/60)} minutes; time left: {int(time_elapsed*(fraction_left/fraction_done)/60)} minutes")
    
    #return top_anns    
    
def main():
    
    layer_num = 1
    train_method = "sgd"
    number_neurons_to_try = range(1, 110, 5)
    activations_to_try = ["linear", "relu", "elu", "sigmoid", "softsign", "tanh"] 
    ann_type = "FFN"
    train = 20
    test = 20
    exploreANNs(layer_num, train_method, number_neurons_to_try, activations_to_try, ann_type, train, test, 0)

if __name__ == "__main__":
    main()
    