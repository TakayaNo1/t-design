import numpy as np

def generate_label(data_type, size):
    """ Create labels for different types of data.
        The Haar measure makes 0, all others make 1.
        Args:
            data_type := String specifying data type(haar, clif, ...)
            size      := Size of label to be created(int)
        Return:
            label := 1d ndarray with the size "size", which elements are 0 or 1
    """
    if data_type == "haar":
        label = np.zeros(size, dtype=np.int8)
    elif data_type == "clif" or data_type == "lrc" or data_type == "rdc":
        label = np.ones(size, dtype=np.int8)
    else:
        print("**CAUTION** Detect undefined data_type :", data_type)
        print("label 1 is created.")
        label = np.ones(size, dtype=np.int8)
    
    return label

def scaler_to_label(output):
    """ Function to convert a scalar to a label {0,1} when the output of model.predict()
        in linear regression or NN model is a scalar instead of a label.
        Arg:
            output := the output of model.predict()
        Return:
            ndarray := 1d list with label 0 or 1 as an element
    """
    predicted_label = [0 if i < 0.5 else 1 for i in output.flatten()]
    return np.array(predicted_label)