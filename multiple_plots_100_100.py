import os
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

def make_plots(anns_folders):
    
    basic_datapath = os.path.abspath(os.getcwd()) #+os.sep+ f"results_{loss_function}"
    categories = ["category 1", "category 2", "category 3", "category 4", "category 5", "category 6", "controllable load"]
    
    for category in range(7):
        f1 = plt.figure()
        ann_id = 0
        for ann in anns_folders:
            
            datapath_read =  basic_datapath +os.sep+ ann +os.sep+ "abs_errors.csv"
            error = []
            with open(datapath_read) as f:
                
                reader = csv.reader(f)
                for row in reader:
                    error.append(float(row[category]))
                    
                    
            ############################################################
            #[adapted from https://www.tutorialspoint.com/how-to-plot-cdf-in-matplotlib-in-python]
            count, bins_count = np.histogram(error, bins=100)
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            plt.plot(bins_count[1:], cdf, label = f"ANN {ann_id+1}")
            ann_id += 1
            ############################################################
        border = [0.9 for _ in range(110)]
        plt.plot(border, "k--")
        
        plt.grid()
        plt.legend(loc="upper right")
        plt.xlabel("absolute error [p.u.]")
        plt.ylabel("Probability")
        plt.xlim([0, 0.3])
        plt.ylim([0-0.1, 1+0.1])
        path_to_save = basic_datapath +os.sep+ "plots" +os.sep+ f"{categories[category]} (winners).png"
        plt.savefig(path_to_save, bbox_inches='tight')
        plt.close()
        print(f"plot for category {category} done")
    

def main():
    winners = ["results_100_100" +os.sep+ "place_1_20_20",
               "results_100_100" +os.sep+ "place_2_20_20",
               "results_100_100" +os.sep+ "place_1_20_100",
               "results_100_100" +os.sep+ "place_2_20_100",
              ]    
    
    make_plots(winners)
    
    
if __name__ == "__main__":
    main()