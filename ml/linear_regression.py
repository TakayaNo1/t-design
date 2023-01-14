from util import data_loader, label_generator

import datetime
import glob
import os
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


def sklearn_lr(shuffle=True):
    """Function to perform linear regression using scikit-learn library.
       We want to perform a binary classification, so we classify from the output of the regression.
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
    ##make a directory path to save linear regression results
    ##(e.g. datasets/dataset1/LR/20210729123234/)
    dir_path = "datasets" + SEP + data_name + SEP + "LR" + SEP + dt_index + SEP
    ##Create a directory to store the results
    os.makedirs(dir_path)

    ##Read the dataset
    train_data, train_label, valid_data, valid_label, test_dataset, test_labelset \
        = data_loader.load_data("LR", data_name, dt_index)

    if shuffle:
        ##set the seed
        seed = np.random.randint(2**31)
        ##Combine training and validation data and divide randomly
        num_train_data = train_data.shape[0]
        temp_data = np.concatenate([train_data, valid_data], axis=0)
        temp_label = np.concatenate([train_label, valid_label], axis=0)
        from sklearn.model_selection import train_test_split
        train_data, valid_data, train_label, valid_label = train_test_split(temp_data, temp_label, train_size=num_train_data, random_state=seed)

    ## define a model set to do normalization
    lr = LinearRegression(normalize=True, n_jobs=-1)

    ##start training
    print("\r" + "fitting...", end="")
    start = time.perf_counter()
    lr.fit(train_data, train_label)
    finish = time.perf_counter()

    ##Output time spent on training
    print("\r" + "fit time : {}[s]".format(finish-start))

    ##Make a directory to save the labels of the predictions output by the regression model.
    label_path = dir_path + "predicted_labels" + SEP 
    os.makedirs(label_path)
    ##Prepare a text file to save the training results(discrimination accuracy) of linear regression.
    result_file = open(dir_path+"result.txt", mode="w")

    ##Get predictive labels for training data
    predicted_train = label_generator.scaler_to_label(lr.predict(train_data))
    ##Calculate identification accuracy(TP/(TP+FP)) and output
    train_acc = accuracy_score(train_label, predicted_train)
    print("\ntrain :", train_acc)
    ##Save identification accuracy and identification results
    result_file.write("train : {}\n".format(train_acc))
    np.save(label_path+"train_predicted_label", predicted_train)

    ##Get predictive labels for validation data
    predicted_valid = label_generator.scaler_to_label(lr.predict(valid_data))
    ##Calculate identification accuracy(TP/(TP+FP)) and output
    valid_acc = accuracy_score(valid_label, predicted_valid)
    print("valid :", valid_acc)
    ##Save identification accuracy and identification results
    result_file.write("valid : {}\n".format(valid_acc))
    np.save(label_path+"valid_predicted_label", predicted_valid)

    ##result of test data
    for i in range(len(test_dataset)):
        ##Get predictive labels for test data
        predicted_test = label_generator.scaler_to_label(lr.predict(test_dataset[i]))
        ##Calculate identification accuracy(TP/(TP+FP)) and output
        test_acc = accuracy_score(test_labelset[i], predicted_test)
        print("test{} : {}".format(i+1, test_acc))
        ##Save identification accuracy and identification results
        result_file.write("test{} : {}\n".format(i+1, test_acc))
        np.save(label_path+"test{}_predicted_label".format(i+1), predicted_test)
    
    ##Close text file to save linearSVM results
    result_file.close()

    ##Save parameters, etc.
    with open(dir_path+"paras.txt", mode="w") as f:
        f.write("fit time : {}\n".format(finish-start))
        f.write("shuffle  : {}\n".format(shuffle))
        if shuffle:
            f.write("seed     : {}\n".format(seed))

if __name__ == "__main__":
    sklearn_lr(shuffle=True)