def get_num_qubits(data_name):
    """ Reads and returns the value of the number of qubits from a file that saves
        the parameters used to create the teacher data.
        Arg:
            data_name(string) := Folder name string for the dataset
        Return:
            Nq(int) := Number of qubits in data creation
    """
    with open("./datasets/{}/info.txt".format(data_name), mode="r") as f:
        l_strip = [s.strip() for s in f.readlines()]
        for line in l_strip:
            if "Nq :" in line:
                chars = line.split(":")
                Nq = int(chars[-1])
                break
    
    return Nq

def get_birCorr_moment(dir_path):
    """ Get the number of bit correlations calculated and the order of the moment
        Arg:
            dir_path(string)  := String for the name of the folder where the experiment results will be saved.
        Returns:
            k(int)             := Number of bits of correlation calculated
            k_prime_list(list) := List whose elements are numerical values(int) of the order of the calculated moments
    """

    with open(dir_path+"extract_parameters.txt", mode="r") as f:
        l_strip = [s.strip() for s in f.readlines()]
        for line in l_strip:
            if "k :" in line:
                chars = line.split(":")
                k = int(chars[-1])
            
            elif "k` :" in line:
                chars = line.split(":")[-1]
                moment_list = chars[2:-1].split(",")
                k_prime_list = [int(i) for i in moment_list]

    return k, k_prime_list