This project will compare performance of different ANNs (FFN, CNN, LSTM and hybrid ones) for demand decomposition.
In the resopritory there is a file runner.py. it allows a user to specity a type of ANN (curently is can be either FFN or LSTM), number of neurons of hidden layers as an array and files to train and test.
The next thing that will be added is a algorithm that will search for the best ANN. A grid seach will be applied. Initially it was planned to try every possibility, but with this method time will rise exponentialy with number of layers. Thus, a different approach will be implemented. Search will start with one network and then will change hyperparameters in one layer without changing remaining layers. This will be done for all layers. To increase performance the proccess will be repeated several times 
