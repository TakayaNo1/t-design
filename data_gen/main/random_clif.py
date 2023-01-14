from group.clifford_group_circuit import CliffordCircuitGroup
from group.clifford_group import CliffordGroup
from qulacs import QuantumState
from qulacs.gate import CNOT, DenseMatrix
import itertools
import time
import numpy as np
import datetime
import sys
import multiprocessing

def input_parameters():
    """Function that takes parameters input from the command line, stores them in a dictionary type, and returns
    """
    paras_dict = {}

    ##input number of data you want to create(|S|)
    print("input number of data (|S|) :", end=(" "))
    paras_dict["S"] = int(input())
    print("Nu :", end=(" "))
    paras_dict["Nu"] = int(input())
    print("Ns :", end=(" "))
    paras_dict["Ns"] = int(input())
    print("Nq :", end=(" "))
    paras_dict["Nq"] = int(input())
    print("Local random clifford?(1:=Yes, 0:=No) :", end=(" "))
    local_in = int(input())
    if local_in == 1:
        paras_dict["local"] = 1
        print("input circuit depth :", end=(" "))
        paras_dict["depth"] = int(input())
        print("CNOT & 1q Clifford?(1:=Yes, 0:=No) :", end=(" "))
        paras_dict["CNOT_1qC"] = int(input())
    else:
        paras_dict["local"] = 0

    return paras_dict

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

def sim_local_random_clifford(S, Nu, Ns, Nq, depth, RU_index_list, comb_list, np_seed):
    ##Set the seed of numpy to be OS independent
    np.random.seed(np_seed)

    ##Define the array of teacher data to be created finally
    teacher_data = np.empty((S, len(comb_list)*20), dtype=np.float32)
    ##Prepare an array to store the results of the measurement probability(expected value of bit correlation) calculations
    MP_list = np.empty((Nu, len(comb_list)), dtype=np.float32)
    ##Preparation of quantum state
    state = QuantumState(Nq)
    ## Declare 2qubit Clifford group, the rank is 11520
    ## Gate columns are returned instead of 4x4 matrices
    ccg = CliffordCircuitGroup(2)
    ##order = ccg.order <= 11520
    #cg = CliffordGroup(2)

    ##Create a list of binary numbers
    binary_num_list = np.empty((2**Nq, Nq), dtype=np.int8)
    for i in range(2**Nq):
        binary_num = np.array(list(bin(i)[2:].zfill(Nq))).astype(np.int8)
        binary_num[binary_num == 0] = -1 #convert 0 to -1
        binary_num_list[i] = binary_num

    for i in range(S):
        ##Fix initial state per teacher data, this is the seed for that
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ##Set initial state to Haar random state
            state.set_Haar_random_state(haar_seed)
            ##Apply 2-qubit random cliffords to each other
            for k in range(1, depth+1):
                for qubit_index in RU_index_list[k%2]:
                    circuit = ccg.get_element(np.random.randint(11520))
                    ccg.simulate_circuit_specific_qubit(2, circuit, state, qubit_index)
                    #DenseMatrix([qubit_index, qubit_index+1], cg.get_element(np.random.randint(11520))).update_quantum_state(state)
            ##perform a measurement
            result_bin = np.array([binary_num_list[i] for i in state.sampling(Ns)])
            ##Calculate bit correlations and compute measurement probabilities
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]].copy()
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[j][k] = np.mean(bit_corr)
        ##Calculate the moment of measument probability
        teacher_data[i] = np.array([np.power(MP_list, m).mean(axis=0) for m in range(1, 21)]).flatten()
        print("\r{} / {} finished...".format(i+1, S), end=(""))
    print("")
    
    return teacher_data

def sim_local_random_clif_CNOTand1qubitClif(S, Nu, Ns, Nq, depth, RU_index_list, comb_list, np_seed):
    ##Set the seed of numpy to be OS independent
    np.random.seed(np_seed)

    ##Define the array of teacher data to be created finally
    teacher_data = np.empty((S, len(comb_list)*20), dtype=np.float32)
    ##Prepare an array to store the results of the measurement probability(expected value of bit correlation) calculations
    MP_list = np.empty((Nu, len(comb_list)), dtype=np.float32)
    ##Set initial state to Haar random state
    state = QuantumState(Nq)
    ## Declare 2qubit Clifford group, the rank is 24
    cg = CliffordGroup(1)
    ##order = cg.order <= 24

    ##Create a list of binary numbers
    binary_num_list = np.empty((2**Nq, Nq), dtype=np.int8)
    for i in range(2**Nq):
        binary_num = np.array(list(bin(i)[2:].zfill(Nq))).astype(np.int8)
        binary_num[binary_num == 0] = -1 #convert 0 to -1
        binary_num_list[i] = binary_num

    for i in range(S):
        ##Fix initial state per teacher data, this is the seed for that
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ##Set initial state to Haar random state
            state.set_Haar_random_state(haar_seed)
            ##Apply 2-qubit random cliffords to each other
            for k in range(1, depth+1):
                for qubit_index in RU_index_list[k%2]:
                    CNOT(qubit_index, qubit_index+1).update_quantum_state(state)
                    clif_matrix = cg.sampling(2)
                    DenseMatrix(qubit_index, clif_matrix[0]).update_quantum_state(state)
                    DenseMatrix(qubit_index+1, clif_matrix[1]).update_quantum_state(state)
            ##perfomr a measurement
            result_bin = np.array([binary_num_list[i] for i in state.sampling(Ns)])
            ##Calculate bit correlations and compute measurement probabilities
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]].copy()
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[j][k] = np.mean(bit_corr)
        ##Calculate the moment of measument probability
        teacher_data[i] = np.array([np.power(MP_list, m).mean(axis=0) for m in range(1, 21)]).flatten()
        print("\r{} / {} finished...".format(i+1, S), end=(""))
    print("")

    return teacher_data

def sim_random_clifford(S, Nu, Ns, Nq, comb_list, np_seed):
    ##Set the seed of numpy to be OS independent
    np.random.seed(np_seed)

    ##Define the array of teacher data to be created finally
    teacher_data = np.empty((S, len(comb_list)*20), dtype=np.float32)
    ##Prepare an array to store the results of the measurement probability(expected value of bit correlation) calculations
    MP_list = np.empty((Nu, len(comb_list)), dtype=np.float32)
    ##Set initial state to Haar random state
    state = QuantumState(Nq)
    ## Declare Nq-qubit Clifford group
    ## Gate columns are returned instead of (2^n)x(2^n) matrices
    ccg = CliffordCircuitGroup(Nq)
    ##Get the number of elements in the Clifford group
    order = ccg.order

    ##Create a list of binary numbers
    binary_num_list = np.empty((2**Nq, Nq), dtype=np.int8)
    for i in range(2**Nq):
        binary_num = np.array(list(bin(i)[2:].zfill(Nq))).astype(np.int8)
        binary_num[binary_num == 0] = -1 #convert 0 to -1
        binary_num_list[i] = binary_num

    for i in range(S):
        ##Fix initial state per teacher data, this is the seed for that
        haar_seed = np.random.randint(2147483648) #2^31
        for j in range(Nu):
            ##Set initial state to Haar random state
            state.set_Haar_random_state(haar_seed)
            #state.set_Haar_random_state(1746904691)
            #state.set_zero_state()
            ##apply a Nq-qubit random clifford operator
            circuit = ccg.get_element(gen_random_index(order))
            ccg.simulate_circuit(Nq, circuit, state)
            ##perfomr a measurement
            result_bin = np.array([binary_num_list[i] for i in state.sampling(Ns)])
            ##Calculate bit correlations and compute measurement probabilities
            for k, combination in enumerate(comb_list):
                bit_corr = result_bin[:, combination[0]].copy()
                for index in range(1, len(combination)):
                    bit_corr *= result_bin[:, combination[index]]
                MP_list[j][k] = np.mean(bit_corr)
        ##Calculate the moment of measument probability
        teacher_data[i] = np.array([np.power(MP_list, m).mean(axis=0) for m in range(1, 21)]).flatten()
        #teacher_data[i] = np.array([np.abs(np.power(MP_list, m)).mean(axis=0) for m in range(1, 21)]).flatten()
        print("\r{} / {} finished...".format(i+1, S), end=(""))
    print("")
    
    return teacher_data


def main(n_proc = -1, **kwargs):
    ## read parameters
    if len(kwargs) == 0:
        paras_dict = input_parameters()
    else:
        paras_dict = kwargs
    ##List of indexes of qubit during bit correlation calculation
    comb_list = gen_comb_list(paras_dict["Nq"])

    ##Obtain the time when the process starts
    start = time.perf_counter()

    ##Determine the number of processes to generate
    if n_proc == -1:
        n_proc = multiprocessing.cpu_count()

    ## Calculation of Local Random Clifford
    if paras_dict["local"] == 1:
        RU_index_list = []
        ## Index of qubit over which the local random clifford is applied when the depth is "odd"
        RU_index_list.append([i for i in range(1, paras_dict["Nq"]-1, 2)])
        ## Index of qubit over which the local random clifford is applied when the depth is "even"
        RU_index_list.append([i for i in range(0, paras_dict["Nq"]-1, 2)])

        ##parallel execution
        if n_proc != 1:
            multi_S, remain = divmod(paras_dict["S"], n_proc)
            print("S={}, n_proc={}, multi_S={}or{}"
                  .format(paras_dict["S"], n_proc, multi_S+1, multi_S))
            args = [(multi_S+1, paras_dict["Nu"], paras_dict["Ns"],paras_dict["Nq"],
                     paras_dict["depth"], RU_index_list, comb_list, np.random.randint(2147483648))
                    if i < remain else
                     (multi_S, paras_dict["Nu"], paras_dict["Ns"],paras_dict["Nq"],
                     paras_dict["depth"], RU_index_list, comb_list, np.random.randint(2147483648))
                    for i in range(n_proc)]
            ##start parallel execution by calling the function parallely
            if paras_dict["CNOT_1qC"] == 0:
                p = multiprocessing.Pool(n_proc)
                returns = p.starmap(sim_local_random_clifford, args)
                p.close()
            elif paras_dict["CNOT_1qC"] == 1:
                p = multiprocessing.Pool(n_proc)
                returns = p.starmap(sim_local_random_clif_CNOTand1qubitClif, args)
                p.close()
            ##Consolidate the results of each thread's execution
            result = np.concatenate(returns, axis=0)

        ##sequential execution
        else:
            ##Quantum circuit simulation and calculation of measurement probabilities, sequentially
            if paras_dict["CNOT_1qC"] == 0:
                result = sim_local_random_clifford(paras_dict["S"],  paras_dict["Nu"], paras_dict["Ns"], paras_dict["Nq"],
                                                   paras_dict["depth"], RU_index_list, comb_list, np.random.randint(2147483648))
            elif paras_dict["CNOT_1qC"] == 1:
                result = sim_local_random_clif_CNOTand1qubitClif(paras_dict["S"],  paras_dict["Nu"], paras_dict["Ns"], paras_dict["Nq"],
                                                                 paras_dict["depth"], RU_index_list, comb_list, np.random.randint(2147483648))

    ##Calculation of Random Cliford operator
    else:
        if n_proc != 1:
            multi_S, remain = divmod(paras_dict["S"], n_proc)
            print("S={}, n_proc={}, multi_S={}or{}"
                  .format(paras_dict["S"], n_proc, multi_S+1, multi_S))
            args = [(multi_S+1, paras_dict["Nu"], paras_dict["Ns"],paras_dict["Nq"], comb_list, np.random.randint(2147483648))
                    if i < remain else
                    (multi_S, paras_dict["Nu"], paras_dict["Ns"],paras_dict["Nq"], comb_list, np.random.randint(2147483648))
                    for i in range(n_proc)]
            ##start parallel execution by calling the function parallely 
            p = multiprocessing.Pool(n_proc)
            returns = p.starmap(sim_random_clifford, args)
            p.close()
            ##Consolidate the results of each thread's execution
            result = np.concatenate(returns, axis=0)            

        else:
            ##Quantum circuit simulation and calculation of measurement probabilities, sequentially
            result = sim_random_clifford(paras_dict["S"],  paras_dict["Nu"], paras_dict["Ns"], 
                                         paras_dict["Nq"], comb_list, np.random.randint(2147483648))
    
    ##Obtain the time at the end of processing
    finish = time.perf_counter()
    ##Obtain the current time
    dt_now = datetime.datetime.now()
    dt_index = dt_now.strftime("%Y%m%d%H%M%S")
    ##save the simulaiton result
    np.save("../result/clif_{}.npy".format(dt_index), result)
    ##save the all parameters
    with open("../result/info_clif_{}.txt".format(dt_index), mode="w") as f:
        f.write("|S| : {}\n".format(paras_dict["S"]))
        f.write(" Nu : {}\n".format(paras_dict["Nu"]))
        f.write(" Ns : {}\n".format(paras_dict["Ns"]))
        f.write(" Nq : {}\n".format(paras_dict["Nq"]))
        if paras_dict["local"] == 1:
            f.write("depth : {}\n".format(paras_dict["depth"]))
            f.write("CNOT&1q clif : {}\n".format(paras_dict["CNOT_1qC"]))
        f.write("bit corrlation : {}\n".format(paras_dict["Nq"]))
        f.write("dim of moments : 1~20\n")
    ##Various output
    print('\nData is saved as "clif_{}.npy".'.format(dt_index))
    print('Information(parameters) of this data is in "info_clif_{}.npy".'.format(dt_index))
    print("Creating Time : {}[s].".format(finish-start))


if __name__ == "__main__":
    ## Number of parallel processes for parallel computation during Clifford simulation.
    ## The maximum number of threads that can be used when -1, and defaults to -1.
    n_proc = -1
    ##Specify the number of processes to be created with a command line argument such as "n_proc=1"
    args = sys.argv
    for arg in args:
        if "n_proc=" in arg:
            n_proc = int(arg.split("=")[-1])
    
    main(n_proc)