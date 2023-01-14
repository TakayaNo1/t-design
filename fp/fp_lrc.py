from fp_code import FP_calc_base
import numpy as np
from scipy.stats import unitary_group


class FP_LRC(FP_calc_base.FP_main_base):
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001, patience=5, monitor="mean") -> None:
        super(FP_LRC, self).__init__(Nq=Nq, depth=depth, t=t, epsilon=epsilon, patience=patience, monitor=monitor)
        self.circ_type = "LRC"
        ##Calculate the number of 2-qubit local Haar gates in each layer of the LRC
        self.num_gates_depthOdd = Nq // 2
        if self.Nq % 2 == 1:
            self.num_gates_depthEven = Nq // 2
        else:
            self.num_gates_depthEven = (Nq // 2) - 1

    def sample_U(self) -> np.ndarray:
        ##Identity matrix with the size (2^Nq)*(2*Nq).
        ##Finally, the desired unitary is created by computing the matrix product
        ##of the combined local 2-qubit haar gates at each depth.
        big_unitary = np.identity(2**self.Nq) 

        ##Repeat for circuit depth
        for i in range(self.depth, 0, -1):
            ##if the depth is odd
            if i % 2 == 1:
                ##The top gate is a 2-qubit haar random gate
                small_unitary = unitary_group.rvs(4)
                ##Take as many tensor products as needed for "small_unitary" and combine local 2-qubit haar gate
                for _ in range(self.num_gates_depthOdd - 1):
                    small_unitary = np.kron(small_unitary, unitary_group.rvs(4))
                ##When both the number of qubits and the depth of the circuit are odd, the Identity is combined to the last(bottom qubit)
                if self.Nq % 2 == 1:
                    small_unitary = np.kron(small_unitary, np.identity(2))
            ##if the depth is even
            else:
                ##Whenever the circuit depth is even, the first is always Identity
                small_unitary = np.identity(2)
                ##Take as many tensor products as needed for "small_unitary" and combine local 2-qubit haar gate
                for _ in range(self.num_gates_depthEven):
                    small_unitary = np.kron(small_unitary, unitary_group.rvs(4))
                ##If the number of qubits is even and the depth of the circuit is even, put Identity at the end.
                if self.Nq % 2 == 0:
                    small_unitary = np.kron(small_unitary, np.identity(2))
            ##Merging "num_qubits" size of unitaries created for each depth by applying
            big_unitary = big_unitary @ small_unitary
        
        return big_unitary
    
    def calculate(self) -> None:
        print("Now => circ:{}, Nq:{}, depth:{}, t:{}".format(self.circ_type, self.Nq, self.depth, self.t))
        super(FP_LRC, self).calculate()
        
def main():
    Nq = 7
    depth = 11
    t = 4
    epsilon = 0.001
    monitor = "mean"
    patience = 5
    FP_LRC_calclator = FP_LRC(Nq=Nq, depth=depth, t=t, epsilon=epsilon, monitor=monitor, patience=patience)
    FP_LRC_calclator.calculate()
    FP_LRC_calclator.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(Nq, depth, t))

    """
    FP_LRC_calclator = FP_LRC()
    Nq_list = [i for i in range(3, 8)]
    depth_list = [i for i in range(4, 21)]
    t_list = [i for i in range(2, 6)]
    epsilon = 0.1
    FP_LRC_calclator.set_parameter("epsilon", 0.1)
    for t in t_list:
        FP_LRC_calclator.set_parameter("t", t)
        for Nq in Nq_list:
            FP_LRC_calclator.set_parameter("Nq", Nq)
            for depth in depth_list:
                FP_LRC_calclator.set_parameter("depth", depth)
                FP_LRC_calclator.calculate()
                FP_LRC_calclator.save_result(foldername="LRC_Nq{}_depth{}_t{}".format(Nq, depth, t))
    """

if __name__ == "__main__":
    main()