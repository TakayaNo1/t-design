from util import data_loader, ElementSearch

import datetime
import glob
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def sklearn_rf(shuffle=True):
    """Function to perform random forest using scikit-learn library.
       Arg:
           shuffle := bool type flag to split train and valid data randomly(default=True)
    """
    ##Get path delimiter ("/" or "\") depending on OS
    SEP = os.sep
    ##Get the current time as a string (e.g.) July 28, 2021 18:40:39 => 20210728184039
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    ##set the seed
    seed = np.random.randint(2**31)
    
    ##input the filename
    print("input dataset : ", end=(""))
    data_name = input()
    if not os.path.isdir("datasets" + SEP + data_name):
        print('ERROR: Cannnot find the dataset "{}"'.format(data_name))
        return -1
    ##make a directory path to save random forest results
    ##(e.g. datasets/dataset1/RF/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "RF" + SEP + dt_index + SEP
    ##Create a directory to store the results
    os.makedirs(dir_path)

    ##Read the dataset
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("RF", data_name, dt_index)
    
    if shuffle:
        ##Combine training and validation data and divide randomly
        num_train_data = train_data.shape[0]
        temp_data = np.concatenate([train_data, valid_data], axis=0)
        temp_label = np.concatenate([train_label, valid_label], axis=0)
        from sklearn.model_selection import train_test_split
        train_data, valid_data, train_label, valid_label = train_test_split(temp_data, temp_label, train_size=num_train_data, random_state=seed)

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

    ##define a model
    forest = RandomForestClassifier(random_state=seed, n_jobs=-1)

    ##start training
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    forest.fit(train_data, train_label)
    finish = time.perf_counter()

    ##Output time spent on training
    print("\r" + "fit time : {}[s]".format(finish-start))

    ##Make a directory to save the labels of the predictions outpuy by RandomForest
    label_path = dir_path + "predicted_labels" + SEP 
    os.makedirs(label_path)
    ##Prepare a text file to save the training results(discrimination accuracy) of RandomForest
    result_file = open(dir_path+"result.txt", mode="w")

    ##Output the result
    ##Results of training data
    train_acc = forest.score(train_data, train_label)
    print("\ntrain :", train_acc)
    result_file.write("train : {}\n".format(train_acc))
    np.save(label_path+"train_predicted_label", forest.predict(train_data))

    ##Results of validation data
    valid_acc = forest.score(valid_data, valid_label)
    print("valid :", valid_acc)
    result_file.write("valid : {}\n".format(valid_acc))
    np.save(label_path+"valid_predicted_label", forest.predict(valid_data))

    ##Results of test data
    for i in range(len(test_dataset)):
        test_acc = forest.score(test_dataset[i], test_labelset[i])
        print('test{} : {}'.format(i+1, test_acc))
        result_file.write("test{} : {}\n".format(i+1, test_acc))
        np.save(label_path+"test{}_predicted_label".format(i+1), forest.predict(test_dataset[i]))
    
    ##Close text file to save RandomForest results
    result_file.close()

    ##Save parameters, etc.
    with open(dir_path+"paras.txt", mode="w") as f:
        f.write("shuffle  : {}\n".format(shuffle))
        f.write("seed     : {}\n".format(seed))
        f.write("fit time : {}\n".format(finish-start))

    ##Get the importance of feature vectors
    importances = forest.feature_importances_
    ##Sort by importance(descending order) and get the list of indices at that time
    indices = np.argsort(-importances)
    
    ##Save index and coefficients(importance) to csv file
    ##Saved in the order of the index, so it is easier to see if they are sorted.
    Searcher.search_and_save_all(indices, importances)


if __name__ == "__main__":
    sklearn_rf()