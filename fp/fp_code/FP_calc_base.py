from .FP_base import FPBase
import numpy as np
import os

class FP_main_base(FPBase):
    """ Abstract (base) class for calculating FramePotential.
        It is defined by inheriting from the class "FPBase" in "FP_base.py".
        Since there may be various patterns of circuits such as LRC and RDC that you want to calculate,
        you can create a new class that only samples unitary by inheriting from this abstract class.
    """
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001, patience=5, monitor="mean") -> None:
        """ constructor
        """
        super(FP_main_base, self).__init__(Nq=Nq, depth=depth, t=t, epsilon=epsilon)
        self.monitor = monitor
        if monitor == "mean":
            self.patience = patience #Variable for how many times to monitor the loop
            self.para_dict = {"patience" : patience} #Dictionary type for saving parameters in file writing
            self.result_arr = np.empty(2) #2d ndarray for saving the calcuation results
            """It is 1d now, but as the calculation proceeds, we increase the first dimension as (1,2)->(2,2)->...->(n,2).
               The three elements of the second dimension are explained as follows:
                   [x,1] := List of results when computing "Tr|UV^{\daggar}|^{2t} "
                   [x,2] := list of values for the average of the Frame Potential
            """

        elif monitor == "std":
            self.result_arr = np.empty(3) #2d ndarray for saving the calcuation results
            """It is 1d now, but as the calculation proceeds, we increase the first dimension as (1,3)->(2,3)->...->(n,3).
               The three elements of the second dimension are explained as follows:
                   [x,1] := List of results when computing "Tr|UV^{\daggar}|^{2t} "
                   [x,2] := list of values for the average of the Frame Potential
                   [x,3] := list of values for the standard deviation of the Frame Potential
            """
            ##The value of the mean of the squares of the results of a single-shot calculation used to calculate the standard deviation
            self.squared = 0.0
            self.para_dict = {} #Dictionary type for saving parameters in file writing. If "std", become empty

        else:
            print('WARNING: monitor target "{}" is invalid. Default monitor "mean" is set.')
            self.monitor = "mean"
            self.patience = patience
            self.para_dict = {"patience" : patience}
            self.result_arr = np.empty(2)
    
    def set_parameter(self, key, val) -> None:
        if key == "monitor":
            if val == "mean":
                self.monitor == val
                self.result_arr = np.empty(2)
                self.para_dict = {"patience" : val}
            elif val == "std":
                self.monitor == val
                self.result_arr = np.empty(3)
                self.para_dict = {}
                self.squared = 0.0
            else:
                print('WARNING: The parameter "monitor" is specified, but value "{}" is invalid.'.format(val))
                print('         Valid paramters are "mean" or "std". Now monitor is "{}"'.format(self.monitor))
        
        elif key == "patience":
            self.patience = val

        else:
            super(FP_main_base, self).set_parameter(key, val)
        
    def calculate(self) -> None:
        """ A method that actually calculates until convergence.
            Repeat "randomly sample unitary matrix => compute" until the value of standard deviation
            is less than or equal to the value "epsilon" specified as a parameter.
        """
        if self.monitor == "mean":
            ##Remove from the next while loop during the first one and the patience times and calculate them first.
            ##Putting it inside increases the number of conditional decisions in the if statement and slows it down.
            count = self.patience + 1
            for i in range(count):
                U = self.sample_U()
                Vdag = np.conjugate(self.sample_U().T)
                potential = np.abs(np.trace(U@Vdag))**(2*self.t)
                if i == 0:
                    ##The first value of average is the same as the single result
                    self.result_arr = np.array([potential, potential]).reshape(1,2)
                else:
                    temp = np.append(self.result_arr[:,0], potential)
                    self.result_arr = np.vstack( (self.result_arr, [potential, np.mean(temp)]) )

            while True:
                U = self.sample_U() #sample U randomly
                Vdag = np.conjugate(self.sample_U().T) #sample V randomly and transpose, complex conjugation
                potential = np.abs(np.trace(U@Vdag))**(2*self.t) #calculate the frame potential
                count += 1
                #print("\r  {} times calculated...".format(count), end="") #output the number of calculation
                if self.check_mean_convergence(potential): #convergence judgment
                    print("")
                    break
                ##output result sequntialy(it is faster not to do)
                print("\r  [{:.3f}, {:.3f}] ({} times calculated)".format(self.result_arr[count-1][0], self.result_arr[count-1][1], count) + "   ", end="")
                if count % 10000 == 0: #When the number of calculations is a multiple of 10000
                    ##Save the results of calculations in progress
                    self.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(self.Nq, self.depth, self.t))
                """
                ##forced termination
                if count == 10000:
                    print("")
                    print("avg:{}, std:{}".format(np.mean(self.result_arr[:,0]), np.std(self.result_arr[:,0])))
                    break
                """

        elif self.monitor == "std":
            ##Remove from the next while loop during the first one and the patience times and calculate them first.
            ##Putting it inside increases the number of conditional decisions in the if statement and slows it down.
            for i in range(2):
                U = self.sample_U()
                Vdag = np.conjugate(self.sample_U().T)
                potential = np.abs(np.trace(U@Vdag))**(2*self.t)
                if i == 0:
                    ##The first value of average is the same as the single result.
                    ##The first standard deviation is 0.0 for now (this value will not be used).
                    self.result_arr = np.array([potential, potential, 0.0]).reshape(1,3)
                elif i == 1:
                    ## The second average is calculated normally
                    ## The second standard deviation is also calculated normally
                    temp = [self.result_arr[0][0], potential]
                    self.result_arr = np.vstack( (self.result_arr, 
                                                [potential, np.mean(temp), np.std(temp)])
                                            )
                    self.squared = np.mean([j**2 for j in temp]) #Compute the value of the average of the squares of the elements

            ##Start calculation of following while loop from the third time until convergence
            count = 2 #Counter showing the number of calculations, since it has already been calculated twice.
            while True:
                U = self.sample_U() #sample U randomly
                Vdag = np.conjugate(self.sample_U().T) #sample V randomly and transpose, complex conjugation
                potential = np.abs(np.trace(U@Vdag))**(2*self.t) #calculate the frame potential
                count += 1
                #print("\r  {} times calculated...".format(count), end="") #output the number of calculation
                if self.check_std_convergence(potential): #convergence judgment
                    print("")
                    break
                ##output result sequntialy(it is faster not to do)
                print("\r  [{:.3f}, {:.3f}, {:.3f}] ({} times calculated)".format(self.result_arr[count-1][0], self.result_arr[count-1][1], self.result_arr[count-1][2], count) + "   ", end="")
                if count % 10000 == 0: #When the number of calculations is a multiple of 10000
                    ##Save the results of calculations in progress
                    self.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(self.Nq, self.depth, self.t))
                """
                ##forced termination
                if count == 10000:
                    print("")
                    print("avg:{}, std:{}".format(np.mean(self.result_arr[:,0]), np.std(self.result_arr[:,0])))
                    break
                """

    def check_mean_convergence(self, fp_shot) -> bool:
        """ Method to determine if the values of Frame Potentail is converged.
            It calculates the average value and returns a flag at the end according to the condition.
            Argument:
                fp_shot(float) := Single-shot calculation result of Frame Potential
            Return:
                flag(bool) := Converged if True, not converged if False
        """
        ##Calculate the value of the average
        num_calculated = self.result_arr.shape[0]
        new_avg = (self.result_arr[-1][1] * num_calculated + fp_shot) / (num_calculated+1)
        ##Save results of single-shot calculations and averages
        self.result_arr = np.vstack( (self.result_arr, [fp_shot, new_avg]) )

        ## Compares the latest calculation result with all calculation results up to the patience cycle,
        ## and returns "True" if all results are less than or equal to epsilon.
        for i in range(self.patience):
            if np.abs(self.result_arr[-1][1] - self.result_arr[-1-(i+1)][1]) > self.eps:
                return False
        return True

    def check_std_convergence(self, fp_shot) -> bool:
        """ Method to determine if the values of Frame Potentail is converged.
            It calculates the average value and returns a flag at the end according to the condition.
            Argument:
                fp_shot(float) := Single-shot calculation result of Frame Potential
            Return:
                flag(bool) := Converged if True, not converged if False
        """
        ##Calculate the value of the average
        num_calculated = self.result_arr.shape[0]
        new_avg = (self.result_arr[-1][1] * num_calculated + fp_shot) / (num_calculated+1)
        ##Calculate the value of the standard deviation
        self.squared = (self.squared * num_calculated + fp_shot**2) / (num_calculated+1)
        new_std = np.sqrt(self.squared - new_avg**2)
        ##Save the results of single-shot calculations and mean and standard deviation results
        self.result_arr = np.vstack( (self.result_arr, [fp_shot, new_avg, new_std]) )

        ##Returns a flag indicating whether converged or not depending on the value of the standard deviation
        if new_std < self.eps:
            return True
        else:
            return False

    def save_result(self, foldername="", log=True) -> None:
        """ Method to write the parameter "patience", just add an argument to save_result in the class "FPBase" and call it
        """
        super(FP_main_base, self).save_result(foldername=foldername, log=log, paras=self.para_dict)