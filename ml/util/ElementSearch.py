from . import read_parameter
import itertools
from scipy.special import comb

def gen_comb_list(num_qubits):
    """Function which returns patterns of all combinations.
       The return value is a 2d list, where the first dimension
       is the number of patterns and the second dimension is the actual pattern.
       Example) if 3-qubit, then
           [[0], [1], [2],       <= 1-bit correlation
           [0,1], [0,2], [1,2],  <= 2-bit correlation
           [0,1,2]]              <= 3-bit correlation
    """
    bit_list = [i for i in range(num_qubits)]
    result = []

    for i in range(1, num_qubits+1):
        for j in itertools.combinations(bit_list, i):
            result.append(list(j))
    
    return result

class Element_Searcher():
    """PCA and RF provide an index of the dimension of the feature vector of highest importance among the teacher data.
       The data for this index is determined by how many point bit correlations and what order of moment they are.
       However, if the moments are not taken in sequence, the correct value is not returned.
       OK => [1,2,3,4] or [4,5,6,7,8], NG => [1,3,5,7,9] or [4,6,8,10]
    """
    def __init__(self, data_name, dir_path):
        """ constructer
        """
        ##Position and moment order of the bit correlation you wish to find from the specified dimension
        self.bitcorr = []
        self.moment = 0
        self.dir_path = dir_path
        ##Parameter acquisition
        Nq = read_parameter.get_num_qubits(data_name)
        k, self.k_prime_list = read_parameter.get_birCorr_moment(dir_path)
        ##Get bit correlation list
        self.bitcorr_list = gen_comb_list(Nq)
        ##Calculate from "Nq" and "k" how many bit correlations are being sought to how many points and
        ##where the bits are located at that time
        self.corr_num = 0
        for i in range(k):
            self.corr_num += comb(Nq, i, exact=True)
        if Nq != k:
            self.bitcorr_list = self.bitcorr_list[:self.corr_num]
    
    def search(self, dim):
        """calculate from specified dimension
        """
        self.bitcorr = self.bitcorr_list[dim % self.corr_num]
        self.moment = self.k_prime_list[(dim // self.corr_num)]
    
    def output(self):
        """Output the position of the bit correlation and the order of the moment to be obtained
        """
        print("bitcorr:{}, k':{}".format(self.bitcorr, self.moment))
    
    def search_and_save_all(self, dims, coef, filename=""):
        """Calculate from specified dimension.
           Results are saved to a tsv file (horizontally tab-delimited).
        """
        ##Open a file to save the results
        if len(filename) == 0:
            ##If no filename is specified, save the file as "fv_components.tsv"
            result_file = open(self.dir_path+"fv_components.tsv", mode="w")
        else:
            ##If no file name is specified, save the file as "[filename].tsv"
            result_file = open(self.dir_path+filename+".tsv", mode="w")

        for dim in dims:
            self.bitcorr = self.bitcorr_list[dim % self.corr_num]
            self.moment = self.k_prime_list[(dim // self.corr_num)]
            result_file.write("{}\t{}\t{}\n".format(self.bitcorr, self.moment, coef[dim]))
        
        result_file.close()