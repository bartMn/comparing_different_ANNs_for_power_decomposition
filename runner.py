import ann
import os

PATH = os.path.abspath(os.getcwd())

def train_save(model, train_method, model_to_load = None):
    model.create_ann(train_method)

def train_save_pred(model, train_method, model_to_load = None):
    model.create_ann(train_method)
    #model.plot_cdf_pdf(PATH, 3)
    model.write_predictions_to_csv(first_prediction = 10081, number_of_predictions = 1440)

def load_pred(model, train_method, model_to_load):
    model.load_model(model_to_load)
    model.plot_cdf_pdf(PATH, 3)
    #model.write_predictions_to_csv(first_prediction = 9601, number_of_predictions = 3000)

def main():
    
    #setting up a model
    ann_type = "LSTM"               #type of ANN
    layers = [11]               #array of numbers of units in each layer 
    house_train = 20                
    house_test = 100
    train_method = "adam"
    files_to_train = [f"total_P_new{house_train}", f"total_Q_new{house_train}", f"decomposition_new{house_train}"]  #files that are used to train ANN
    files_to_test =  [f"total_P_new{house_test}", f"total_Q_new{house_test}", f"decomposition_new{house_test}"]     #files that are used to make predictions
    file_to_write = "predictions"           #file to save predictions
    model_to_load = f"test_model{ann_type}" #file to load a model (only relevant if you want to load), make sure you load the same type (e.g. FFN)
    steps = 150                              #number of steps (only relevant for LSTM)
    
    model = ann.ANN(ann_type = ann_type, layers = layers, files_to_train= files_to_train, files_to_test= files_to_test, file_to_write = file_to_write, steps = steps)
    model.add_testing_data(files_to_test)
    actions = [train_save, train_save_pred, load_pred]      #array of function pointers
    
    print("what do you want?")
    print("1. make, train and save a model")
    print("2. make, train, save and make predictions")
    print("3. load a model and make predictions")
    action = int(input("enter a number: "))
    #action = 2
    
    try:
        #calling a desired function
        #the solution mimics c-like swith-case
        #time to get to desired function is O(1)
        #elif-s would make developing code more difficult and time would be O(n)
        actions[action-1](model, train_method, model_to_load)
    
    except IndexError:
        print("PROGRAM TERMINATED")
        print("Enter one of the allowed numbers next time")
        
    #model = ann.ANN(ann_type = ann_type, layers = layers, files_to_train= files_to_train, files_to_test= files_to_test, steps = steps)
    #model.write_predictions_to_csv(first_prediction = 5876, number_of_predictions = 5000, file_to_write = file_to_write_predictions)


if __name__ == "__main__":
    main()