from util import data_loader, read_parameter, ElementSearch

import datetime
import glob
import os
import time
import copy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def draw_train_base(label_exist, axes, train_info, train_data, train_label):
    """ We want to overlay a plot of validation or test data on top of a plot of training data.
        This function plots the training data in that case. 
        Args:
            label_exist(int)                  := Whether to plot with labels or not, 0 for not
            axes(class of axes of matplotlib) := empty axes object
            train_info(list)                  := List to remember the type of data and the number of data for that teacher data
            train_data(ndarray)               := Principal component vector of training data
            train_label(ndarray)              := Prediction labels for training data(default=0)
    """
    p = 0 #Pointer to list position
    for data_type, data_size in train_info:
        if data_type == "haar":
            color = "blue" #set Haar-train as blue
        elif data_type == "clif":
            color = "red" #set Clif-train as red
        else:
            color = "black" #set others as black
        
        if label_exist == 0:
            axes.scatter(train_data[p:p+data_size,0], train_data[p:p+data_size,1], label=data_type+"-train", marker='o', alpha=0.5, s=15, c=color)
        else:
            ##Extract labels from one of the combined teacher data
            partial_label = train_label[p:p+data_size]
            ##List of feature vectors to save vectors predicted to be 0(haar) or 1(clif), respectively
            pred_0, pred_1 = [], []
            ##Separate each list with list comprehension notation
            [pred_0.append(val) if partial_label[i] == 0 else pred_1.append(val) for i,val in enumerate(train_data[p:p+data_size])]
            ##Draw in as a scatter plot
            if len(pred_0) != 0:
                pred_0 = np.array(pred_0)
                axes.scatter(pred_0[:,0], pred_0[:,1], label=data_type+"-train:pred0", marker='o', alpha=0.5, s=15, c=color) #0と予測されたデータは○でプロット
            if len(pred_1) != 0:
                pred_1 = np.array(pred_1)
                axes.scatter(pred_1[:,0], pred_1[:,1], label=data_type+"-train:pred1", marker='*', alpha=0.5, s=15, c=color) #1と予測されたデータは☆でプロット
        ##Draw a histogram
        p += data_size #advance a pointer


def sklearn_pca(repeat=True):
    """ Function to perform principle component analysis using scikit-learn library.
        Arg:
            repeat := bool type flag to display graphs, etc. (default=False)
    """
    ##Get path delimiter ("/" or "\") depending on OS
    SEP = os.sep

    ##Get the current time as a string (e.g.) July 28, 2021 18:40:39 => 20210728184039
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")

    ##input the filename
    print("input dataset : ", end=(""))
    data_name = input()
    if not os.path.isdir("datasets" + SEP + data_name):
        print('ERROR: Cannnot find the dataset "{}"'.format(data_name))
        return -1
    ##make a directory path to save PCA results
    ##(e.g. datasets/dataset1/PCA/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "PCA" + SEP + dt_index + SEP
    ##Create a directory to store the results
    os.makedirs(dir_path)

    ##Search for folders where predictive labels obtained by running machine learning algorithms are saved
    label_path = sorted(glob.glob("datasets/"+data_name+"/**/predicted_labels", recursive=True))

    ##Select which of the predicted labels to use when plotting
    if len(label_path) == 0:
        print("\nPredicted labels were not found.")
        label_choice = 0
    else:
        print("\nChoose the label to plot picture :")
        print("  0 := not use")
        for i,path in enumerate(label_path):
            print("  {} := {}".format(i+1, (path.split(data_name+SEP)[-1]).split(SEP+"predicted_labels")[0]))
        print("\nlabel_choice : ", end="")
        label_choice = int(input())

    ##When using labels for predictions
    if label_choice != 0:
        ##Obtain the number of bit correlations and the order of the moments when machine learning is performed
        ##and predictions are made, and regenerate the teacher data with these parameters.
        k, kp_list = read_parameter.get_birCorr_moment(label_path[label_choice-1].split("predicted_labels")[0])
        train_data, train_info, valid_data, valid_info, test_dataset, test_infoset = data_loader.load_data("PCA", data_name, dt_index, k, kp_list)
        ##Loading Predicted Labels
        train_pred_label, valid_pred_label, test_pred_labelset = data_loader.load_pred_labels(label_path[label_choice-1])
    ##When PCA is performed without label information
    else:
        ##Input the number of bit correlations and the order of the moments to read the dataset
        train_data, train_info, valid_data, valid_info, test_dataset, test_infoset = data_loader.load_data("PCA", data_name, dt_index)
        """
        ##Combine training and validation data and divide randomly
        temp_data = np.concatenate([train_data, valid_data], axis=0)
        #temp_label = np.concatenate([train_label, valid_label], axis=0)
        from sklearn.model_selection import train_test_split
        seed = np.random.randint(2**31)
        print("seed :", seed)
        train_data, valid_data = train_test_split(temp_data, train_size=0.75, random_state=seed)
        """
        ##When calling "draw_train_base()", the variable "train_pred_label" is passed as an argument,
        ##and the information in "train_pred_label" is used to determine the number of histograms, so make it 0.
        train_pred_label = np.zeros(train_data.shape[0], dtype=int)

    ##Scale to the mean become 0, the variance become 1 for each feature in the teacher data
    scaler = StandardScaler()
    ##Calculate shift and scaling ratios using train data and transform
    train_data = scaler.fit_transform(train_data)
    ##validation and test data are transformed with calculated shift and scaling ratios from training dataset
    valid_data = scaler.transform(valid_data)
    test_dataset = [scaler.transform(test_data) for test_data in test_dataset]

    ##Class for finding out how many point bit correlations were computed at what order of moments for a feature
    ##from the index of the feature vector when the part of the feature vector with the largest contribution is output.
    Searcher = ElementSearch.Element_Searcher(data_name, dir_path)
    
    ##Reduce the dimensionality to be the variance after projection max, and holds the first and second principal components
    pca = PCA(n_components=2, svd_solver="full")

    ##start PCA
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    pca.fit(train_data)
    finish = time.perf_counter()

    ##Output time spent on fitting
    print("\r" + "fit time : {}[s]".format(finish-start))

    ##Perform PCA on each of the data
    train_reduc = pca.transform(train_data)
    valid_reduc = pca.transform(valid_data)
    testset_reduc = [pca.transform(test_data) for test_data in test_dataset]

    """Draw a scatter plot of the training data with the first principal component(horizontal axis)
       and the second principal component(vertical axis). The above figure is a scatterplot and the
       below figure is a histogram of the density of the first principal component.
    """
    fig = plt.figure() #making figure
    ax1 = fig.add_subplot(2,1,1) #Plot principal components onto the above part in the fig
    ax2 = fig.add_subplot(2,1,2) #plot histogram of the density onto the below part in the fig
    hist_bins = train_pred_label.shape[0] // 20 #the number of bar of histgram
    ##Plot the training data, which had been created by combining them, one at a time in sequence.
    p = 0 #Pointer to list position
    for data_type, data_size in train_info:
        if data_type == "haar":
            color = "blue" #set Haar-train as blue
        elif data_type == "clif":
            color = "red" #set clif-train as red
        else:
            color = "black" #set others as black
        
        if label_choice == 0:
            ##Plot with color coding and symbol "o" when there is no prediction label
            ax1.scatter(train_reduc[p:p+data_size, 0], train_reduc[p:p+data_size, 1], label=data_type+"-train", marker='o', alpha=0.5, s=15, c=color)
        
        else:
            ##Extract labels from one of the combined teacher data
            partial_label = train_pred_label[p:p+data_size]
            ##List of feature vectors to save vectors predicted to be 0(haar) or 1(clif), respectively
            pred_0, pred_1 = [], []
            ##Separate each list by list comprehension notation
            [pred_0.append(val) if partial_label[i] == 0 else pred_1.append(val) for i,val in enumerate(train_reduc[p:p+data_size])]
            ##Draw in as a scatter plot
            if len(pred_0) != 0:
                pred_0 = np.array(pred_0)
                ax1.scatter(pred_0[:,0], pred_0[:,1], label=data_type+"-train:pred0", marker='o', alpha=0.5, s=15, c=color) #Data predicted to be 0 are plotted with a circle
            if len(pred_1) != 0:
                pred_1 = np.array(pred_1)
                ax1.scatter(pred_1[:,0], pred_1[:,1], label=data_type+"-train:pred1", marker='*', alpha=0.5, s=15, c=color) #Data predicted to be 1 are plotted with a star
        ##Draw a histogram
        ax2.hist(train_reduc[p:p+data_size, 0], label=data_type, bins=hist_bins, density=True, alpha=0.8, color=color)
        p += data_size #advance a pointer
    ##Display the name and title of each axis of the two graphs, as well as grid lines and legends
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("A plot of training data")    
    ax1.legend()
    ax1.grid()
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("Density")
    ax2.set_title("Histgram of PC1 (sum=1)")
    ax2.legend()
    ax2.grid()
    fig.set_tight_layout(True)
    plt.savefig(dir_path+"train.png")
    if not repeat:
        plt.show()
    
    """ Plot the training and validation data
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    draw_train_base(label_choice, ax, train_info, train_reduc, train_pred_label)
    ##Plot the training data, which had been created by combining them, one at a time in sequence.
    p = 0 #Pointer to list position
    for data_type, data_size in valid_info:
        if data_type == "haar":
            color = "yellow" #set Haar-valid as yellow
        elif data_type == "clif":
            color = "green" #set clif-valid as green
        else:
            color = "black" ##set others as blue
        
        if label_choice == 0:
            ##Plot with color coding and symbol "o" when there is no prediction label
            ax.scatter(valid_reduc[p:p+data_size, 0], valid_reduc[p:p+data_size, 1], label=data_type+"-valid", marker='o', alpha=0.5, s=15, c=color)
        
        else:
            ##Extract labels from one of the combined teacher data
            partial_label = valid_pred_label[p:p+data_size]
            ##List of feature vectors to save vectors predicted to be 0(haar) or 1(clif), respectively
            pred_0, pred_1 = [], []
            ##Separate each list by list comprehension notation
            [pred_0.append(val) if partial_label[i] == 0 else pred_1.append(val) for i,val in enumerate(valid_reduc[p:p+data_size])]
            ##Draw in as a scatter plot
            if len(pred_0) != 0:
                pred_0 = np.array(pred_0)
                ax.scatter(pred_0[:,0], pred_0[:,1], label=data_type+"-valid:pred0", marker='o', alpha=0.5, s=15, c=color) #Data predicted to be 0 are plotted with a circle
            if len(pred_1) != 0:
                pred_1 = np.array(pred_1)
                ax.scatter(pred_1[:,0], pred_1[:,1], label=data_type+"-valid:pred1", marker='*', alpha=0.5, s=15, c=color) #Data predicted to be 1 are plotted with a star
        p += data_size #advance a pointer
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("A plot of training and validiation data")    
    ax.legend()
    ax.grid()
    fig.set_tight_layout(True)
    plt.savefig(dir_path+"valid.png")
    if not repeat:
        plt.show()

    """ Plot the training and test data
    """
    for test_num, each_test_info in enumerate(test_infoset):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        draw_train_base(label_choice, ax, train_info, train_reduc, train_pred_label)
        ##Plot the training data, which had been created by combining them, one at a time in sequence.
        p = 0 #Pointer to list position
        for data_type, data_size in each_test_info:
            ##Decide the color of the graph, except for Haar, which is green and should be changed if it is included in the test data set.
            if data_type == "haar":
                color = "yellow" #set Haar-test as yellow
            elif data_type == "clif":
                color = "green" #set clif-test as green
            elif data_type == "lrc":
                color = "green" #set lrc-test as green
            elif data_type == "rdc":
                color = "green" #set rdc-test as green

            if label_choice == 0:
                ##Plot with color coding and symbol "o" when there is no prediction label
                ax.scatter(testset_reduc[test_num][p:p+data_size, 0], testset_reduc[test_num][p:p+data_size, 1], label=data_type+"-test{}".format(test_num+1), marker='o', alpha=0.5, s=15, c=color)
            
            else:
                ##Select the target from a list of test data and labels, and then select a range to extract the data
                partial_data = testset_reduc[test_num][p:p+data_size]
                partial_label = test_pred_labelset[test_num][p:p+data_size]
                ##List of feature vectors to save vectors predicted to be 0(haar) or 1(clif), respectively
                pred_0, pred_1 = [], []
                ##Separate each list by list comprehension notation
                [pred_0.append(val) if partial_label[i] == 0 else pred_1.append(val) for i, val in enumerate(partial_data)]
                ##Draw in as a scatter plot
                if len(pred_0) != 0:
                    pred_0 = np.array(pred_0)
                    ax.scatter(pred_0[:,0], pred_0[:,1], label=data_type+"-test{}:pred0".format(test_num+1), marker='o', alpha=0.5, s=15, c=color) #Data predicted to be 0 are plotted with a circle
                if len(pred_1) != 0:
                    pred_1 = np.array(pred_1)
                    ax.scatter(pred_1[:,0], pred_1[:,1], label=data_type+"-test{}:pred1".format(test_num+1), marker='*', alpha=0.5, s=15, c=color) #Data predicted to be 1 are plotted with a star
            
            p += data_size #advance a pointer

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("A plot of training and test{} data".format(test_num+1))
        ax.legend()
        ax.grid()
        fig.set_tight_layout(True)
        plt.savefig(dir_path+"test{}.png".format(test_num+1))
        if not repeat:
            plt.show()

    ##Save parameters, etc.
    with open(dir_path+"paras.txt", mode="w") as f:
        if label_choice == 0:
            f.write("used label : None\n")
        else:
            f.write("used label : {}\n".format((label_path[label_choice-1].split(data_name+SEP)[-1]).split(SEP+"predicted_labels")[0]))


    ##Get the importance of feature vectors
    importances = pca.components_
    ##Sort by importance(descending order) and get the list of indices at that time
    indices = np.argsort(-importances)
    
    ##Save index and coefficients(importance) to csv file
    ##Saved in the order of the index, so it is easier to see if they are sorted.
    Searcher.search_and_save_all(indices[0], importances[0], filename="components_PC1")
    Searcher.search_and_save_all(indices[1], importances[1], filename="components_PC2")


if __name__ == "__main__":
    sklearn_pca()