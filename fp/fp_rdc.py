from fp_code import FP_calc_base
import numpy as np


class FP_RDC(FP_calc_base.FP_main_base):
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001, patience=5, monitor="mean") -> None:
        super(FP_RDC, self).__init__(Nq=Nq, depth=depth, t=t, epsilon=epsilon, patience=patience, monitor=monitor)
        self.circ_type = "RDC"
        ##Create a matrix of Hadmard gates over n-qubits which applied after the diagonal matrix
        hadmard = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        self.hadmard_n_size = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        for _ in range(Nq-1):
            self.hadmard_n_size = np.kron(self.hadmard_n_size, hadmard)

    def sample_U(self) -> np.ndarray:
        ##Identity matrix with the size (2^Nq)*(2*Nq).
        ##Finally, the desired unitary is created by computing the matrix product
        ##of the combined local 2-qubit haar gates at each depth.
        big_unitary = np.identity(2**self.Nq) 

        ##Repeat for circuit depth
        for _ in range(self.depth):
            big_unitary = big_unitary @ self.hadmard_n_size @ np.diag(np.exp(np.random.rand(2**self.Nq) * 2 * np.pi * 1j))
        
        return big_unitary
    
    def calculate(self) -> None:
        print("Now => circ:{}, Nq:{}, depth:{}, t:{}".format(self.circ_type, self.Nq, self.depth, self.t))
        super(FP_RDC, self).calculate()
        
def main():
    Nq = 5
    depth = 6
    t = 2
    epsilon = 0.001
    monitor = "mean"
    patience = 5
    FP_RDC_calclator = FP_RDC(Nq=Nq, depth=depth, t=t, epsilon=epsilon, monitor=monitor, patience=patience)
    FP_RDC_calclator.calculate()
    FP_RDC_calclator.save_result(foldername="RDC_Nq{}_depth{}_t{}".format(Nq, depth, t))

    """
    FP_RDC_calclator = FP_RDC()
    Nq_list = [i for i in range(3, 8)]
    depth_list = [i for i in range(4, 21)]
    t_list = [i for i in range(2, 6)]
    epsilon = 0.1
    FP_RDC_calclator.set_parameter("epsilon", 0.1)
    for t in t_list:
        FP_RDC_calclator.set_parameter("t", t)
        for Nq in Nq_list:
            FP_RDC_calclator.set_parameter("Nq", Nq)
            for depth in depth_list:
                FP_RDC_calclator.set_parameter("depth", depth)
                FP_RDC_calclator.calculate()
                FP_RDC_calclator.save_result(foldername="RDC_Nq{}_depth{}_t{}".format(Nq, depth, t))
    """

if __name__ == "__main__":
    main()