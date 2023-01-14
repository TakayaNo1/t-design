from util import data_loader

import datetime
import glob
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def sklearn_linear_svm(shuffle=True):
    """Function to perform linear support vector machine using scikit-learn library.
       Arg:
           shuffle := bool type flag to split train and valid data randomly(default=True)
    """
    ##Get path delimiter ("/" or "\") depending on OS
    SEP = os.sep
    ## Get the current time as a string (e.g.) July 28, 2021 18:40:39 => 20210728184039
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    
    ##input the filename
    print("input dataset : ", end=(""))
    data_name = input()
    if not os.path.isdir("datasets" + SEP + data_name):
        print('ERROR: Cannnot find the dataset "{}"'.format(data_name))
        return -1
    ##make a directory path to save linear_svm results
    ##(e.g. datasets/dataset1/LinearSVM/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "LinearSVM" + SEP + dt_index + SEP
    ##Create a directory to store the results
    os.makedirs(dir_path)

    ##Read the dataset
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("LinearSVM", data_name, dt_index)
    
    if shuffle:
        ##set the seed
        seed = np.random.randint(2**31)
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

    ##define a model
    """ Ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC  
        dual: bool, default=True
            Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features.
    """
    clf = LinearSVC(dual=False)

    ##start training
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    clf.fit(train_data, train_label)
    finish = time.perf_counter()

    ##Output time spent on training
    print("\r" + "fit time : {}[s]".format(finish-start))
    
    ##Make a directory to save the labels of the predictions outpuy by LinearSVM
    label_path = dir_path + "predicted_labels" + SEP 
    os.makedirs(label_path)
    ##Prepare a text file to save the training results(discrimination accuracy) of LinearSVM
    result_file = open(dir_path+"result.txt", mode="w")

    ##Output the result
    ##Results of training data
    train_acc = clf.score(train_data, train_label)
    print("\ntrain :", train_acc)
    result_file.write("train : {}\n".format(train_acc))
    np.save(label_path+"train_predicted_label", clf.predict(train_data))

    ##Results of validation data
    valid_acc = clf.score(valid_data, valid_label)
    print("valid :", valid_acc)
    result_file.write("valid : {}\n".format(valid_acc))
    np.save(label_path+"valid_predicted_label", clf.predict(valid_data))
    
    ##Results of test data
    for i in range(len(test_dataset)):
        test_acc = clf.score(test_dataset[i], test_labelset[i])
        print('test{} : {}'.format(i+1, test_acc))
        result_file.write("test{} : {}\n".format(i+1, test_acc))
        np.save(label_path+"test{}_predicted_label".format(i+1), clf.predict(test_dataset[i]))
    
    ##Close text file to save LinearSVM results
    result_file.close()

    ##Save parameters, etc.
    with open(dir_path+"paras.txt", mode="w") as f:
        f.write("dual     : False\n")
        f.write("fit time : {}\n".format(finish-start))
        f.write("shuffle  : {}\n".format(shuffle))
        if shuffle:
            f.write("seed     : {}\n".format(seed))


if __name__ == "__main__":
    sklearn_linear_svm()