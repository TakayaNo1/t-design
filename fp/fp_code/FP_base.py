import numpy as np
import os

class FPBase():
    """ The most basic abstract (base) class for calculating Frame Potential.
        It defines basic operations such as setting and storing parameters
        such as the number of qubits and the depth of the circuit.
    """
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001) -> None:
        """ constructor
        """
        self.Nq = Nq        #number of qubit
        self.depth = depth  #Depth of quantum circuit(or None for circuits with no depth)
        self.t = t          #the order of t-design 
        self.eps = epsilon  #Variables used in the convergence decision
    
    def set_parameter(self, key, val) -> None:
        if key == "Nq":
            self.Nq = val
        elif key == "depth":
            self.depth = val
        elif key == "t":
            self.t = val
        elif key == "epsilon":
            self.eps = val
        else:
            print('Error: key "{}" is invalid, defalut parameter is set.')

    def calculate(self) -> None:
        raise NotImplementedError("This is abstract method")

    def sample_U(self) -> np.ndarray:
        raise NotImplementedError("This is abstract method")

    def check_mean_convergence(self, fp_shot) -> bool:
        raise NotImplementedError("This is abstract method")
    
    def check_std_convergence(self, fp_shot) -> bool:
        raise NotImplementedError("This is abstract method")

    def save_result(self, foldername="", log=True, paras={}) -> None:
        """ A method to save the results.
            The final result and parameters are saved in a txt file, and the
            calculation process is saved or not according to the argument flag "log".
            Arguments:
                foldername(str) := Name of the folder where the calculation results will be stored.
                                   If empty, the current time.
                log(bool)       := When True, all past calculation processes are saved in csv,
                                   when False, they are not saved.
                paras(dict)     := Dictionary type to hold parameters that you want to store additionally.
                                   When saving, it is written in the form of "{key} : {value}".
        """
        if len(foldername) == 0:
            import datetime
            dt_now = datetime.datetime.now()
            foldername = dt_now.strftime("%Y%m%d%H%M%S")
            
        if not os.path.exists("./result/" + foldername):
            os.makedirs("./result/" + foldername)
        
        with open("./result/" + foldername + "/parameters.txt", mode="w") as f:
            f.write("#qubit  : {}\n".format(self.Nq))
            f.write("depth   : {}\n".format(self.depth))
            f.write("t       : {}\n".format(self.t))
            f.write("epsilon : {}\n".format(self.eps))
            for key in paras.keys():
                f.write("{} : {}\n".format(key, paras[key]))
        
        with open("./result/" + foldername + "/result.txt", mode="w") as f:
            #f.write("{} \pm {}".format(self.result_arr[-1][1], self.result_arr[-1][2]))
            f.write("{} \pm {}".format(self.result_arr[-1][1], np.std(self.result_arr[:][1])))
        
        if log:
            np.savetxt("./result/" + foldername + "/log.csv", self.result_arr, delimiter=",")
