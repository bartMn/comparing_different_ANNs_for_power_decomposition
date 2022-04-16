import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from files import read_data, read_small_data, lstm_data
from models_funcs2 import make_prediction, write_predictions

from sklearn.model_selection import train_test_split
import os
import csv

PATH = os.path.abspath(os.getcwd()) 


class ANN():
    
    #creator function:
    #   ann_type ->       string
    #   layers ->         array of ints, number of nodes in each layer
    #   files_to_train -> array of strings
    #   files_to_test ->  array of strings 
    def __init__(self, ann_type, layers, files_to_train, files_to_test, file_to_write, steps = None):
        
        self.take_smaller_data = True
        self.model_functions = {"FFN" : tf.keras.layers.Dense,
                                "LSTM": tf.keras.layers.LSTM}
        self.type = ann_type
        self.layers = layers
        self.files_to_train = files_to_train
        self.files_to_test = files_to_test
        self.steps = steps
        self.file_to_write = file_to_write
        self.loss_function = "mape"
        #self.loss_function = tf.keras.losses.Huber()
        
        evidence, labels = self.get_data(self.files_to_train)
        
        if self.take_smaller_data:
            evidence = evidence[: 8*1440]
            labels = labels[: 8*1440]
            
        if ann_type == "FFN":
            self.input_type = (2,)
            self.steps = None
            self.x_training, self.x_testing, self.y_training, self.y_testing = train_test_split(
                evidence, labels, test_size=0.125   
                )
            
        elif ann_type == "LSTM":
            self.input_type = (steps, 2)
            evidence, labels = lstm_data(raw_evidence = evidence, raw_labels = labels, seq_lenght = steps)
            
            self.x_training, self.x_testing, self.y_training, self.y_testing = train_test_split(
                evidence, labels, test_size=0.125   
                )
            
            self.x_testing = self.x_testing.reshape((self.x_testing.shape[0], self.x_testing.shape[1], 2))
            self.x_training = self.x_training.reshape((self.x_training.shape[0], self.x_training.shape[1], 2))
            
        elif self.type == "CNN":
            self.input_type = (steps, 2)
            evidence, labels = lstm_data(raw_evidence = evidence, raw_labels = labels, seq_lenght = steps)
            
            self.x_training, self.x_testing, self.y_training, self.y_testing = train_test_split(
                evidence, labels, test_size=0.125   
                )
            
            #self.x_testing2 = self.x_testing2.reshape((self.x_testing2.shape[0], self.x_testing2.shape[1], 2))
            #self.x_training2 = self.x_training2.reshape((self.x_training2.shape[0], self.x_training2.shape[1], 2))
        
        self.model = None
        #self.model = self.create_ann()
        
        
    #gets data for FFN
    def get_data(self, files):
        
        print("read in progress")
        evidence, labels = read_data(files)
        print("data read done")
        return evidence, labels
    
    def add_testing_data(self, files):
        self.files_to_test2 = files
        evidence, labels = self.get_data(files)
        
        if self.take_smaller_data:
            evidence = evidence[: 8*1440]
            labels = labels[: 8*1440]
            
        
        if self.type == "FFN":
            self.input_type = (2,)
            self.steps = None
            self.x_training2, self.x_testing2, self.y_training2, self.y_testing2 = train_test_split(
                evidence, labels, test_size=0.125   
                )
            
        elif self.type == "LSTM":
            self.input_type = (self.steps, 2)
            evidence, labels = lstm_data(raw_evidence = evidence, raw_labels = labels, seq_lenght = self.steps)
            
            self.x_training2, self.x_testing2, self.y_training2, self.y_testing2 = train_test_split(
                evidence, labels, test_size= 1440/len(labels)   #test_size=0.125  
                )
            
            self.x_testing2 = self.x_testing2.reshape((self.x_testing2.shape[0], self.x_testing2.shape[1], 2))
            self.x_training2 = self.x_training2.reshape((self.x_training2.shape[0], self.x_training2.shape[1], 2))
                    
       
       
    def modify_data(self, steps):
        self.steps = steps
        self.add_testing_data(self.files_to_test2)
        
        evidence, labels = self.get_data(self.files_to_train)
        
        if self.take_smaller_data:
            evidence = evidence[: 8*1440]
            labels = labels[: 8*1440]
            
        if self.type == "FFN":
            self.input_type = (2,)
            self.steps = None
            self.x_training, self.x_testing, self.y_training, self.y_testing = train_test_split(
                evidence, labels, test_size=0.125   
                )
            
        elif self.type == "LSTM":
            self.input_type = (steps, 2)
            evidence, labels = lstm_data(raw_evidence = evidence, raw_labels = labels, seq_lenght = steps)
            
            self.x_training, self.x_testing, self.y_training, self.y_testing = train_test_split(
                evidence, labels, test_size= 1440/len(labels)   
                )
            
            self.x_testing = self.x_testing.reshape((self.x_testing.shape[0], self.x_testing.shape[1], 2))
            self.x_training = self.x_training.reshape((self.x_training.shape[0], self.x_training.shape[1], 2))
            
        elif self.type == "CNN":
            self.input_type = (steps, 2)
            evidence, labels = lstm_data(raw_evidence = evidence, raw_labels = labels, seq_lenght = steps)
            
            self.x_training, self.x_testing, self.y_training, self.y_testing = train_test_split(
                evidence, labels, test_size= 1440/len(labels)   
                )
            
      
    #creates, trains, evaluates and saves an ann
    def create_ann(self, train_method):
           	
        # Create a neural network
        model = tf.keras.models.Sequential()

        #first/input layer
        if self.type == "FFN":
            model.add(self.model_functions[self.type](self.layers[0], input_shape= self.input_type, activation="linear"))
            model.add(tf.keras.layers.Dropout(0.4))
            model.add(self.model_functions[self.type](self.layers[1], activation="relu"))
            model.add(tf.keras.layers.Dropout(0.4))
            #model.add(self.model_functions[self.type](self.layers[2], activation="tanh"))
            #model.add(tf.keras.layers.Dropout(0.4))
            
            #model.add(self.model_functions[self.type](self.layers[2], activation="relu"))
            #model.add(tf.keras.layers.Dropout(0.4))
        
            #adding hidden layers
            #for layer in self.layers[1:]:
            #    model.add(self.model_functions[self.type](layer, activation="relu"))
            #    model.add(tf.keras.layers.Dropout(0.4))
        
        
        elif self.type == "LSTM":
            if len(self.layers) > 1:
                model.add(self.model_functions[self.type](self.layers[0], input_shape= self.input_type, activation="relu", return_sequences= True))
                model.add(tf.keras.layers.Dropout(0.4))
            #adding hidden layers
            for layer in self.layers[1:-1]:
                model.add(self.model_functions[self.type](layer, activation="relu", return_sequences= True))
                model.add(tf.keras.layers.Dropout(0.4))
            
            model.add(self.model_functions[self.type](self.layers[-1], activation="relu"))
            model.add(tf.keras.layers.Dropout(0.4))
        
        elif self.type == "CNN":
            print("training CNN")
            model.add(tf.keras.layers.Conv2D(50, (self.steps-1, 1), activation='relu', input_shape=(self.steps, 2, 1)))
            #model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(self.layers[0], activation="relu"))
            model.add(tf.keras.layers.Dropout(0.4))
            #adding hidden layers
            for layer in self.layers[1:]:
                model.add(tf.keras.layers.Dense(layer, activation="relu"))
                model.add(tf.keras.layers.Dropout(0.4))
            
        
        
        #last/output layer    
        model.add(tf.keras.layers.Dense(6, activation="sigmoid"))
        # Train neural network
        model.compile(
            optimizer= train_method,
            loss= self.loss_function,
            #metrics=["mean_squared_logarithmic_error"]
            metrics= [self.loss_function, tf.keras.metrics.CosineSimilarity()]
        )
        
        model.fit(self.x_training, self.y_training, epochs=20)
        # Evaluate how well model performs
        model.evaluate(self.x_testing,  self.y_testing, verbose=2)

        model.save(f"test_model{self.type}")
        print("model saved")
        model.summary()
        self.model = model
        #result3, result4 = self.custom_evaluation()
        
    
    def create_test_ann(self, layers, train_method):
         
        layers = list(layers)   	
        layers_number = int((len(layers)-1)/2)
        neurons = layers[:layers_number]
        activations = layers[layers_number: ]
        output_activation = activations.pop()
        print(f"layers_number: {layers_number}")
        print(f"neurons: {neurons}")
        print(f"activations: {activations}")
        print(f"output_activation: {output_activation}")
        # Create a neural network
        model = tf.keras.models.Sequential()

        #first/input layer
        if self.type == "FFN":
            model.add(self.model_functions[self.type](neurons[0], input_shape= self.input_type, activation= activations[0]))
            model.add(tf.keras.layers.Dropout(0.4))
            #adding hidden layers
            for i in range(1, layers_number):
                model.add(self.model_functions[self.type](neurons[i], activation= activations[i]))
                model.add(tf.keras.layers.Dropout(0.4))
        
        else:
            if len(neurons) > 1:
                model.add(self.model_functions[self.type](neurons[0], input_shape= self.input_type, activation= activations[0], return_sequences= True))
                model.add(tf.keras.layers.Dropout(0.4))
            #adding hidden layers
            for i in range(1, layers_number - 1):
                model.add(self.model_functions[self.type](neurons[i], activation= activations[i], return_sequences= True))
                model.add(tf.keras.layers.Dropout(0.4))
            
            model.add(self.model_functions[self.type](neurons[-1], activation= activations[-1]))
            model.add(tf.keras.layers.Dropout(0.4))
        #last/output layer    
        model.add(tf.keras.layers.Dense(6, activation= output_activation))
        
        # Train neural network
        model.compile(
            optimizer= train_method,
            loss= self.loss_function,
            metrics= [self.loss_function, tf.keras.metrics.CosineSimilarity()]
        )
        
        model.fit(self.x_training, self.y_training, epochs=20)
        # Evaluate how well model performs
        self.model = model
        model.evaluate(self.x_testing,  self.y_testing, verbose=2)

        model.save(f"test_model{self.type}")
        print("model saved")
        model.summary()
        
        result = model.evaluate(self.x_testing,  self.y_testing, verbose=2)
        result2 =  model.evaluate(self.x_testing2,  self.y_testing2, verbose=2)
        
        return result[0], result2[0], result[2], result2[2], 0, 0, 0, 0
        
        
    #this function calculates the sum of average errors and
    #sum of absolute value of errors that are over 0.1
    def custom_evaluation(self, x_test, y_test):
        
        errors_ov_10 = 0
        sum_abs_err_ov10 = 0
        corrupted_rounds = 0
        for i in range(300):
            #print(i)
            #data_in = [x_test[i]]
            #data = np.array(data_in)
            prediction = self.model.predict([x_test[i]])
            #print(prediction[0])
            #temp_sum = 0 
            temp = 0 
            for j in range(6):
                difference = y_test[i][j] - prediction[0][j]
                #temp_sum += difference
                abs_err = abs(difference)
                if abs_err > 0.1:
                    sum_abs_err_ov10 += abs_err
                    errors_ov_10 += 1 
                    temp = 1 
            corrupted_rounds += temp
            #sum_average_error += temp_sum/6
        print("custom evaluation done") 
        #print(sum_average_error, " ", sum_abs_err_ov10)   
        #return sum_average_error, sum_abs_err_ov10
        print(errors_ov_10, " ", sum_abs_err_ov10)   
        return errors_ov_10, sum_abs_err_ov10, corrupted_rounds
        
    def load_model(self, model_to_load):
        self.model = tf.keras.models.load_model(model_to_load)
     
    def plot_cdf_pdf(self, path, data_to_plot):
        
        if data_to_plot == 0:
            self.make_plots(path, 0)
        elif data_to_plot == 1:
            self.make_plots(path, 1)
        else:
            self.make_plots(path, 0)
            self.make_plots(path, 1)
         
    def make_plots(self, path, mode):
        error_data= []
        errors_per_category = [[] for _ in range(6)]
        
        if mode == 0:
            x_test = self.x_testing
            y_test = self.y_testing
            testing_houses = 20
        else:
            x_test = self.x_testing2
            y_test = self.y_testing2
            testing_houses = 100
            
            
        newpath = path + os.sep + f"tested on {testing_houses} houses"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    
            
        path_to_save = newpath +os.sep+ f'predictions.csv'
        with open(path_to_save, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for i in range(len(x_test)):
            #for i in range(10):
            
                if self.type == "FFN":
                    prediction = self.model.predict([x_test[i]]) 
                else:
                    data =np.array([x_test[i]])
                    data = data.reshape((1, self.steps, 2)) 
                    prediction = self.model.predict(data) 
            
                writer.writerow(prediction[0])
                
                difference = [y_test[i][j] - prediction[0][j] for j in range(6)] 
                path_to_save2 = newpath +os.sep+ f'errors.csv'
                with open(path_to_save2, 'a', newline='', encoding='utf-8') as csvfile2:
                    writer2 = csv.writer(csvfile2)
                    writer2.writerow(difference)
                    
        
    #makes one prediction:
    #   element_to_predict -> int
    def make_one_prediction(self, element_to_predict):
        prediction = make_prediction(last_data= element_to_predict, model = self.model, files_to_read = self.files_to_test, steps_num = self.steps)      
        return prediction

    #makes n predictions and writes them to csv file (n= number_of_predictions):
    #   first_prediction ->      int
    #   number_of_predictions -> int
    #   file_to_write ->         string
    def write_predictions_to_csv(self, first_prediction, number_of_predictions):
        write_predictions(first_pred = first_prediction, num_pred = number_of_predictions, model= self.model, files_to_read = self.files_to_test, file_to_write= self.file_to_write, steps= self.steps)
        
    def set_loss_function(self, function):
        self.loss_function = function