from fp_code import FP_calc_base
import sys
sys.path.append("../data_gen/main")
from group import clifford_group
import numpy as np


def gen_random_index(order):
    """ Function to generate and return a random integer in the range [0, order).
        There is np.random.randint that does something similar, but it has an upper
        limit of 2^31, which is smaller than the rank of Clifford group.
    """
    d = str(order)
    dig_num = len(d)
    sig_dig = int(d[0])

    while True:
        clif_index = 0
        a = np.random.randint(sig_dig+1, size=1)
        f = np.append(a, np.random.randint(10, size=dig_num-1))
        clif_index = int("".join(map(str, f)))
        if clif_index < order:
            return clif_index


class FP_RC(FP_calc_base.FP_main_base):
    def __init__(self, Nq=4, depth=5, t=2, epsilon=0.001, patience=5, monitor="mean") -> None:
        super(FP_RC, self).__init__(Nq=Nq, depth=depth, t=t, epsilon=epsilon, patience=patience, monitor=monitor)
        self.circ_type = "RC"
        self.group = clifford_group.CliffordGroup(Nq)
        self.order = self.group.order

    def sample_U(self) -> np.ndarray:
        return self.group.get_element(gen_random_index(self.order))
    
    def calculate(self) -> None:
        print("Now => circ:{}, Nq:{}, depth:{}, t:{}".format(self.circ_type, self.Nq, self.depth, self.t))
        super(FP_RC, self).calculate()
        
def main():
    Nq = 5
    depth = 6
    t = 2
    epsilon = 0.001
    monitor = "mean"
    patience = 5
    FP_RC_calclator = FP_RC(Nq=Nq, depth=depth, t=t, epsilon=epsilon, monitor=monitor, patience=patience)
    FP_RC_calclator.calculate()
    FP_RC_calclator.save_result(foldername="RC_Nq{}_depth{}_t{}".format(Nq, depth, t))

    """
    FP_RC_calclator = FP_RC()
    Nq_list = [i for i in range(3, 8)]
    depth_list = [i for i in range(4, 21)]
    t_list = [i for i in range(2, 6)]
    epsilon = 0.1
    FP_RC_calclator.set_parameter("epsilon", 0.1)
    for t in t_list:
        FP_RC_calclator.set_parameter("t", t)
        for Nq in Nq_list:
            FP_RC_calclator.set_parameter("Nq", Nq)
            for depth in depth_list:
                FP_RC_calclator.set_parameter("depth", depth)
                FP_RC_calclator.calculate()
                FP_RC_calclator.save_result(foldername="RC_Nq{}_depth{}_t{}".format(Nq, depth, t))
    """

if __name__ == "__main__":
    main()