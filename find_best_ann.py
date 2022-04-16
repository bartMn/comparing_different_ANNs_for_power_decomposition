import finding_bestANNv2
import csv


def main():
    
    layer_num = [1,2,3]
    train_method0 = "sgd"
    train_method1 = "adam"
    number_neurons_to_try = range(1, 129, 5)
    activations_to_try = ["linear", "relu", "elu", "sigmoid", "softsign", "tanh"] 
    ann_type = "FFN"
    train = 20
    test = 100
        
    for i in range(3):
       finding_bestANNv2.exploreANNs(layer_num[i], train_method1, number_neurons_to_try, activations_to_try, ann_type, train, test, 0)
                        
###########################################################
    for i in range(3):
        finding_bestANNv2.exploreANNs(layer_num[i], train_method1, number_neurons_to_try, activations_to_try, ann_type, train, test, 1)


if __name__ == "__main__":
    main()