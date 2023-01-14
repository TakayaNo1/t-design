from . import label_generator, read_parameter, ElementSearch

import glob
import os

import numpy as np
import random


def load_data(ml_alg, data_name, dt_index, k=0, kp_list=[]):
    """ Function that reads and returns teacher data, given a folder in the dataset folder.
        The original data is a csv or npy file, with the largest bit correlation and moments
        up to the 20-th order calculated.
        The returns of this function is different for PCA and for machine learning algorithms:
            ・PCA => list of data read
            ・Machine learning system => data molded for machine learning
        Arguments:
            ml_alg(string)    := String of names of machine learning algorithms (e.g. NN, svm, ...)
            data_name(string) := String for dataset folder (e.g. dataset1)
            dt_index(string)  := File name string (e.g., 20210728161623)
            k(int)            := Number of bit correlations to calculate (default=0); if 0, standard input later
            kp_list(list)     := List of orders of the moment to be calculated (default=[]). If empty, standard input later.
        Returns:
            PCA:
                train_data    := Dataset for training (2d ndarray, (number of datasets)*(dimension of features) )
                train_info    := A list with elements listing the type and data size of each of the data
                                 before merging when train_data is created by merging
                valid_data    := Dataset for validation (2d ndarray, (number of datasets)*(dimension of features) )
                valid_info    := A list with elements listing the type and data size of each piece of data before merging to create valid_data.
                test_dataset  := List with test dataset as elements (list with 2-dimensional ndarray as elements)
                test_infoset  := List of lists whose elements are a list of the type and data size of each of the data before merging

            Machine Learning:
                train_data    := Dataset for training (2d ndarray, (number of datasets)*(dimension of features) )
                train_label   := 訓練用のデータのラベル(1次元のndarrayで要素が0か1, 大きさがデータセット数)
                valid_data    := Dataset for validation (2d ndarray, (number of datasets)*(dimension of features) )
                valid_label   := 検証用のデータのラベル(1次元のndarrayで要素が0か1, 大きさがデータセット数)
                test_dataset  := List with test dataset as elements (list with 2-dimensional ndarray as elements)
                test_labelset := List with labels of data for testing as elements (list with 1D ndarray as elements)
    """
    ##Get path delimiter ("/" or "\") depending on OS
    SEP = os.sep

    ##Get the number of qubits
    Nq = read_parameter.get_num_qubits(data_name)
    
    ##Calculate the maximum number of possible combinations when bit correlations are calculated
    Nq_prime = 2 ** Nq - 1

    ##Explore paths in the dataset
    ##Search the "train" folder
    dataset_path = glob.glob('datasets/{}/train/*_*.npy'.format(data_name))
    dataset_path.extend(glob.glob('datasets/{}/train/*_*.csv'.format(data_name)))
    ##search in "valid" folder
    dataset_path.extend(glob.glob('datasets/{}/valid/*_*.npy'.format(data_name)))
    dataset_path.extend(glob.glob('datasets/{}/valid/*_*.csv'.format(data_name)))
    ##search in "test" folder
    dataset_path.extend(glob.glob('datasets/{}/test/**/*_*.npy'.format(data_name), recursive=True))
    dataset_path.extend(glob.glob('datasets/{}/test/**/*_*.csv'.format(data_name), recursive=True))

    ##The order of the lists obtained by "glob" depends on the OS, so sort them to make sure they are the same.
    dataset_path.sort()
    ##get the number of test datasets
    testset_num = len(glob.glob('datasets/{}/test/*'.format(data_name)))
    
    ##Output of the paths that have been explored
    print("Extracted path:")
    print(np.array(dataset_path))
    print("num_testset =", testset_num)

    ##Input the number of bit correlations "k" to be extracted and the order of moments "k'".
    if k == 0:
        ##Selecting and extracting the number of bit correlations is not yet implemented, if needed.
        #print('input k(Max={}) :'.format(Nq), end=(' '))
        #k = int(input())
        k = Nq
        #specific_k_corr_point = np.sort(random.sample([i for i in range(Nq_prime)], Nq_prime//4))
        #print("specific_k_corr_point =", specific_k_corr_point)
    if len(kp_list) == 0:
        print('\ninput k prime')
        print('  input min(min=1) :', end=(' '))
        kp_min = int(input())
        print('  input max(max=20) :', end=(' '))
        kp_max = int(input())
        print('  input step(default=1) :', end=(' '))
        kp_step = int(input())
        #kp_step = 1
        kp_list = [i for i in range(kp_min, kp_max+1, kp_step)]
    print("Dim of moment =", kp_list)
    
    """ A list that holds the data that has been extracted. The list has 4 elements with the number of elements:
          [0] : type of data (haar, clif, lrc, ...)
          [1] : purpose of data (train, valid, test1, ...)
          [2] : feature vector ( (data size)*(2d ndarray of features) )
          [3] : data size of feature vector (int)
    """
    data_list = []
    ##Create a location-specific array to extract only specific parts of the bit correlations
    mom_num = len(kp_list)
    #specific_k_corr_point = np.ravel([specific_k_corr_point*i for i in range(1, mom_num+1)])

    ##Extraction in order
    for data_path in dataset_path:
        ##Output what data is currently being manipulated
        print("\rNow => {}   ".format(data_path), end="")

        ##Identify data type (haar, clif, ...)
        if "haar" in data_path:
            data_type = "haar"
        elif "clif" in data_path:
            data_type = "clif"
        elif "lrc" in data_path:
            data_type = "lrc"
        elif "rdc" in data_path:
            data_type = "rdc"
        
        ##Identify the intended use of the data (train, valid, ...)
        if "train" in data_path:
            purpose = "train"
        elif "valid" in data_path:
            purpose = "valid"
        elif "test" in data_path:
            count = 1
            while True:
                if "test{}".format(count)+SEP in data_path:
                    purpose = "test{}".format(count)
                    break
                else:
                    count += 1
        
        ##Read files according to format
        if "npy" in data_path:
            ##If it is an npy file, it reads normally and with single precision
            data = np.load(data_path).astype(np.float32)
        elif "csv" in data_path:
            ##For csv files, read the delimiter character with a comma.
            data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
            ##Replace csv files with npy(binary) format
            ##When repeating machine learning, npy is faster to load and the file, and filesize is smaller
            np.save(data_path.split(".csv")[0], data)
            os.remove(data_path)
            
        ##Extract the necessary parts
        data = np.concatenate(
            [data[:, Nq_prime*(i-1):(Nq_prime*(i-1))+Nq_prime] for i in kp_list],
            axis=1
        )

        ##only pull out certain parts of the bit correlation
        #data = data[:, specific_k_corr_point] #fancy index

        ##add to list
        #print(data_type, purpose, data.shape[0])
        data_list.append([data_type, purpose, data, data.shape[0]])
    print("\r" + " "*(len(dataset_path[-1])+12) + "\n")
 
    ##Save extracted parameters
    with open("datasets/{}/{}/{}/extract_parameters.txt".format(data_name, ml_alg, dt_index), mode="w") as f:
        f.write('k : {}\nk` : {}\n'.format(k, kp_list))
    
    ##Create data for training. Extract and combine data whose purpose is "train" from the list of data.
    train_data = np.concatenate([temp_list[2] for temp_list in data_list if temp_list[1] == "train"], axis=0)
    ##Create data for validiation. Extract and combine data whose purpose is "valid" from the list of data.
    valid_data = np.concatenate([temp_list[2] for temp_list in data_list if temp_list[1] == "valid"], axis=0)
    ##Create data for testing. Since we want to give more than one set of test data, create them in a list with data sets as elements.
    test_dataset = []
    for i in range(1, testset_num+1):
        ##Create data for testing. Extract and combine data whose purpose is "test*" from the list of data.
        test_dataset.append(np.concatenate([temp_list[2] for temp_list in data_list if temp_list[1] == "test{}".format(i)], axis=0))
        
    if ml_alg == "PCA":
        """ In the case of PCA, labels are either "none" or "predicted by machine learning algorithm", so labels are not created here.
            Instead, create and return a list that holds each data type and data size
        """
        ##For each piece of data that makes up the training data, create a list that holds the type and data size of that data.
        train_info = [[temp_list[0], temp_list[3]] for temp_list in data_list if temp_list[1] == "train"]
        ##For each piece of data that makes up the validation data, create a list that holds the type and data size of that data.
        valid_info = [[temp_list[0], temp_list[3]] for temp_list in data_list if temp_list[1] == "valid"]
        ##For each piece of data that makes up the test data, create a list that holds the type and data size of that data.
        test_infoset = []
        for i in range(1, testset_num+1):
            temp_info = [[temp_list[0], temp_list[3]] for temp_list in data_list if temp_list[1] == "test{}".format(i)]
            test_infoset.append(temp_info)

        return train_data, train_info, valid_data, valid_info, test_dataset, test_infoset

    else:
        """ Create and return labels for machine learning
        """
        ##Create labels for training data. Extract data whose purpose is TRAIN from the list of data, generate labels, and combine them.
        train_label = np.concatenate(
            [ label_generator.generate_label(temp_list[0], temp_list[3]) for temp_list in data_list if temp_list[1] == "train" ],
            axis=0
        )
        ##Create labels for validiation data. Extract data whose purpose is valid from the list of data, generate labels, and combine them.
        valid_label = np.concatenate(
            [ label_generator.generate_label(temp_list[0], temp_list[3]) for temp_list in data_list if temp_list[1] == "valid" ],
            axis=0
        )
        ##Create labels for the test data. Since we want to give more than one set of test data, create them in a list with the data set as an element.
        test_labelset = []
        for i in range(1, testset_num+1):
            ## Create data for testing, extract and combine data whose purpose is "test*" from a list of data.
            temp_label = np.concatenate(
                [ label_generator.generate_label(temp_list[0], temp_list[3]) for temp_list in data_list if temp_list[1] == "test{}".format(i) ],
                axis=0
            )
            test_labelset.append(temp_label)

        return train_data, train_label, valid_data, valid_label, test_dataset, test_labelset


def load_pred_labels(path):
    """ Would like to plot the output of PCA with information on the trained model's output(predictive labels).
        This function that reads the saved predictive labels and returns them along with the information of the included data.
        Arg:
            path(string) := Path of the directory where the predictive labels are stored
        Returns:
            train_label(ndarray) := Prediction labels for training data
            valid_label(ndarray) := Prediction labels for validation data
            test_labelset(list of ndarray) := List with predictive labels for test data as elements
    """
    ##Reading predictive labels for train data
    train_path = glob.glob(path + "/train_*.npy")
    train_label = np.load(train_path[0])

    ##Reading predictive labels for valid data
    valid_path = glob.glob(path + "/valid_*.npy")
    valid_label = np.load(valid_path[0])

    ##Get all paths to npy files labeled test in the specified directory
    test_path = glob.glob(path + "/test*.npy")
    ##want to prepare labes in the orders of test1, test2, test3, ... .
    ##To do this, we want to sort by the number of ** in test**, so we cut out the string and convert the ** to an int type list.
    testnumber_list = [int((i[len(path)+1:].split("_")[0]).split("test")[-1]) for i in test_path]
    ##Get the list of indices when the above list is reordered
    index_order_list = np.argsort(testnumber_list)
    ##Get data in order and add to the list.
    test_label_listset = [np.load(test_path[i]) for i in index_order_list]


    return train_label, valid_label, test_label_listset