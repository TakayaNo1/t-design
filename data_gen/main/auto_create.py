import random_clif
import numpy as np
import sys
import time
import subprocess
import platform
import glob

##Specify the name of the executable file according to your environment
if platform.system() == "Windows":
    env = "win"
    exe_cpp = "msvc_project.exe"
elif platform.system() == "Linux":
    env = "lin"
    exe_cpp = "./main"
else:
    env = "lin"
    exe_cpp = "./main"

def create_one_data(n_proc):
    print("\n*** start creating one data ***")

    ##decide which data you create
    print("\ninput index == 1:haar  2:clifford  3:LRC  4:RDC")
    while True:
        print("You create : ", end=(" "))
        create_index = int(input())
        if create_index == 1:
            ident = "haar"
            break
        elif create_index == 2:
            ident = "clif"
            break
        elif create_index == 3:
            ident = "lrc"
            break
        elif create_index == 4:
            ident = "rdc"
            break
        else:
            print("please input 1, 2, 3, or 4.\n")

    ##input the number of data you want to create(|S|)
    print("input number of data (|S|) :", end=(" "))
    S = int(input())
    print("Nu :", end=(" "))
    Nu = int(input())
    print("Ns :", end=(" "))
    Ns = int(input())
    print("Nq :", end=(" "))
    Nq = int(input())
    depth = 0
    local = 0
    CNOT_1qC = 0
    if create_index == 2 or create_index == 3 or create_index == 4:
        print("input circuit depth :", end=(" "))
        depth = int(input())
        if create_index == 2 and depth != 0:
            local = 1
            print("input CNOT_1qC(1:=Yes, 0:=No) :", end=(" "))
            CNOT_1qC = int(input())

    noise_ope = 0
    noise_prob = 0.0
    if create_index == 3:
        print("input noise operator(0:=nothing, 1:=Depolarizing, 2:=Measurement)", end=(" "))
        noise_ope = int(input())
        if noise_ope == 1 or noise_ope == 2:
            print("input noise probability : ", end=(" "))
            noise_prob = float(input())
    
    if create_index == 2:
        paras = {"S":S, "Nu":Nu, "Ns":Ns, "Nq":Nq, "local":local, "depth":depth, "CNOT_1qC":CNOT_1qC}
        random_clif.main(n_proc, **paras)
        
    if create_index == 1 or create_index == 3 or create_index == 4:
        subprocess.run("{} unitary_type={} S={} Nu={} Ns={} Nq={} depth={} noise_operator={} noise_prob={}" \
                    .format(exe_cpp, create_index-1, S, Nu, Ns, Nq, depth, noise_ope, noise_prob), shell=True)
    
    print("\n\n  ***    All finished!!!    ***\n")
    

def auto_create(n_proc):
    print("\n*** start creating dataset ***")

    ##input folder name and number of test datasets you need
    print("input folder name :", end=(" "))
    folder_name = input()
    print("input the number of test datasets :", end=(" "))
    test_dataset_num = int(input())

    ##set directory pass
    folder_1 = folder_name + "/"  ## folder_1 := "datasetX/"
    folder_2 = folder_1 + "test/" ## folder_2 := "datasetX/test/"

    ##add windows command of making directory
    cmd_array = ["mkdir " + folder_name,            ## mkdir datasetX
                 "mkdir " + folder_1 + "train",     ## mkdir datasetX/trian
                 "mkdir " + folder_1 + "valid",     ## mkdir datasetX/valid
                 "mkdir " + folder_1 + "test",      ## mkdir datasetX/test
                ]

    for i in range(test_dataset_num):
        cmd_array.append("mkdir " + folder_2 + "test{}".format(i+1))  #mkdir datasetX/test/test1

    ##detailed settings for data creation
    print("\nDo you custom? (y/n)")
    while True:
        custom = input()
        if custom == "y" or custom == "n":
            break
        else:
            print('please input "y" or "n".')

    if custom == "y":
        ##decide which data you create
        print("\ninput index == 1:haar  2:clifford  3:LRC  4:RDC (0:don`t create)")
        print("train1 : ", end=(" "))
        train1_type = int(input())
        print("train2 : ", end=(" "))
        train2_type = int(input())
        print("valid1 : ", end=(" "))
        valid1_type = int(input())
        print("valid2 : ", end=(" "))
        valid2_type = int(input())

        test_type = np.empty((test_dataset_num, 2))
        for i in range(test_dataset_num):
            for j in range(2):
                print("test{}-{} : ".format(i+1, j+1), end=(" "))
                test_type[i][j] = int(input())

    elif custom == "n":
        ##decide 2 data type
        print("\ninput index == 1:haar  2:clifford  3:LRC  4:RDC")
        print("data_type1 : ", end=(" "))
        data_type1 = int(input())
        print("data_type2 : ", end=(" "))
        data_type2 = int(input())

    ##summarize the data
    datatype_index = ["haar", "clifford", "LRC", "RDC"]
    order_array = []
    if custom == "y":
        order_array.append(datatype_index[train1_type-1] + " - train")
        order_array.append(datatype_index[train2_type-1] + " - train")
        order_array.append(datatype_index[valid1_type-1] + " - valid")
        order_array.append(datatype_index[valid2_type-1] + " - valid")
        for i in range(test_dataset_num):
            for j in range(2):
                if test_type[i][j] == 0:
                    ##don`t create
                    pass
                else:
                    order_array.append(datatype_index[int(test_type[i][j]-1)] + " - test{}".format(i+1))

    elif custom == "n":
        order_array.append(datatype_index[data_type1-1] + " - train")
        order_array.append(datatype_index[data_type2-1] + " - train")
        order_array.append(datatype_index[data_type1-1] + " - valid")
        order_array.append(datatype_index[data_type2-1] + " - valid")
        for i in range(test_dataset_num):
            order_array.append(datatype_index[data_type1-1] + " - test{}".format(i+1))
            order_array.append(datatype_index[data_type2-1] + " - test{}".format(i+1))

        train1_type = data_type1
        valid1_type = data_type1
        train2_type = data_type2
        valid2_type = data_type2
        test_type = np.empty((test_dataset_num, 2))
        for i in range(test_dataset_num):
            test_type[i][0] = data_type1
            test_type[i][1] = data_type2

    ##input parameters
    print("\ninput parameters of circuit simulation")
    print("train |S| :", end=(" "))
    train_S = int(input())
    print("valid |S| :", end=(" "))
    valid_S = int(input())
    print("each test |S| :", end=(" "))
    test_S = int(input())
    print("Nu :", end=(" "))
    Nu = int(input())
    print("Ns :", end=(" "))
    Ns = int(input())
    print("Nq :", end=(" "))
    Nq = int(input())
    if "clifford" in "".join(order_array) or "LRC" in "".join(order_array) or "RDC" in "".join(order_array):
        ##detailed settings of circuit depth
        print("Do you change the depth according to the data? (y/n)")
        while True:
            depth_custom = input()
            if depth_custom == "y" or depth_custom == "n":
                break
            else:
                print('please input "y" or "n".')

        if depth_custom == "y":
            depth_array = np.zeros(len(order_array), dtype=np.uint32)
            local_array = np.zeros(len(order_array), dtype=np.uint32)
            CNOT_1qC_array = np.zeros(len(order_array), dtype=np.uint32)
            for i in range(len(order_array)):
                if "clifford" in order_array[i]:
                    print("depth({}) : ".format(order_array[i]), end=(" "))
                    depth_array[i] = int(input())
                    if depth_array[i] != 0:
                        local_array[i] = 1
                        print("  CNOT_1qC({}) (1:=Yes, 0:=No): ".format(order_array[i]), end=(" "))
                        CNOT_1qC_array[i] = int(input())
                elif "LRC" in order_array[i] or "RDC" in order_array[i]:
                    print("depth({}) : ".format(order_array[i]), end=(" "))
                    depth_array[i] = int(input())
        elif depth_custom == "n":
            print("depth :", end=(" "))
            i_depth = int(input())
            if i_depth != 0:
                i_local = 1
                print("  CNOT_1qC(1:=Yes, 0:=No) :", end=(" "))
                i_CNOT_1qC = int(input())
            else:
                i_local = 0
                i_CNOT_1qC = 0
    
    if "LRC" in "".join(order_array):
        ##detailed settings of circuit depth
        print("Do you change the noise config according to the data? (y/n)")
        while True:
            noise_custom = input()
            if noise_custom == "y" or noise_custom == "n":
                break
            else:
                print('please input "y" or "n".')
        
        if noise_custom == "y":
            noise_ope_array = np.zeros(len(order_array), dtype=np.uint32)
            noise_prob_array = np.zeros(len(order_array), dtype=np.float32)
            for i in range(len(order_array)):
                if "LRC" in order_array[i]:
                    print("noise operator(0:=nothing, 1:=Depolarizing, 2:=Measurement)({}) : ".format(order_array[i]), end=(" "))
                    noise_ope_array[i] = int(input())
                    if noise_ope_array[i] == 1 or noise_ope_array[i] == 2:
                        print("  noise probability({}) : ".format(order_array[i]), end=(" "))
                        noise_prob_array[i] = float(input())

        elif noise_custom == "n":
            print("noise operator(0:=nothing, 1:=Depolarizing, 2:=Measurement) : ", end=(" "))
            i_noise_ope = int(input())
            if i_noise_ope == 0:
                i_noise_prob = 0.0
            elif i_noise_ope == 1 or i_noise_ope == 2:
                print("  noise probability : ", end=(" "))
                i_noise_prob = float(input())
            

    print("\nYou will create below data")
    print(np.array(order_array))

    print("\nIf all parameters are OK, press Enterkey.")
    input()


    print("start creating data...")
    start = time.perf_counter()
    ##execute windows command of make directory
    for cmd in cmd_array:
        if env == "win":
            subprocess.call(cmd.replace("/", "\\"), shell=True)
        elif env == "lin":
            subprocess.call(cmd, shell=True)

    ##create data
    for i in range(len(order_array)):
        print("\n" + order_array[i])
        ## define data type
        if "haar" in  order_array[i]:
            ident = "haar"
            circuit_id = 1
        elif "clifford" in order_array[i]:
            ident = "clif"
            circuit_id = 2
            if depth_custom == "y":
                i_depth = depth_array[i]
                i_local = local_array[i]
                i_CNOT_1qC = CNOT_1qC_array[i]            
        elif "LRC" in order_array[i]:
            ident = "lrc"
            circuit_id = 3
            if depth_custom == "y":
                i_depth = depth_array[i]
            if noise_custom == "y":
                i_noise_ope = noise_ope_array[i]
                i_noise_prob = noise_prob_array[i]
        elif "RDC" in order_array[i]:
            ident = "rdc"
            circuit_id = 4
            if depth_custom == "y":
                i_depth = depth_array[i]

        ##define the datasize according to the purpose of deeplearning
        if "train" in order_array[i]:
            purpose = "train"
            S = train_S
        elif "valid" in order_array[i]:
            purpose = "valid"
            S = valid_S
        elif "test" in order_array[i]:
            purpose = "test"
            S = test_S

        ##create the data my calling each function
        if circuit_id == 2:
            ##random clifford
            paras = {"S":S, "Nu":Nu, "Ns":Ns, "Nq":Nq, "local":i_local, "depth":i_depth, "CNOT_1qC":i_CNOT_1qC}
            random_clif.main(n_proc, **paras)
        elif circuit_id == 1:
            ##haar measure
            subprocess.run("{} unitary_type=0 S={} Nu={} Ns={} Nq={}".format(exe_cpp, S, Nu, Ns, Nq), shell=True)
        elif circuit_id == 3:
            ##local random circuit
            subprocess.run("{} unitary_type=2 S={} Nu={} Ns={} Nq={} depth={} noise_operator={} noise_prob={}" \
                        .format(exe_cpp, S, Nu, Ns, Nq, i_depth, i_noise_ope, i_noise_prob), shell=True)
        elif circuit_id == 4:
            ##random diagonal circuit
            subprocess.run("{} unitary_type=3 S={} Nu={} Ns={} Nq={} depth={}" \
                        .format(exe_cpp, S, Nu, Ns, Nq, i_depth), shell=True)

        ##Move the created data to correct place
        ##get the path of the data just created now
        if circuit_id == 2:
            datafile_path = glob.glob("../result/clif_*.npy")[-1]
        elif circuit_id == 1 or circuit_id == 3 or circuit_id == 4:
            datafile_path = glob.glob("../result/{}_*.csv".format(ident))[-1]
        infofile_path = glob.glob("../result/info_{}_*.txt".format(ident))[-1]

        ##get the destination path    
        if purpose == "train" or purpose == "valid":
            dest_path = "./" + folder_1 + purpose
        elif purpose == "test":
            for j in range(test_dataset_num):
                if "test{}".format(j+1) in order_array[i].split(" "):
                    testset_num = "test{}".format(j+1)
                    break
            dest_path = "./" + folder_1 + "test/" + testset_num
        
        if env == "win":
            subprocess.run("move {} {}".format(datafile_path, dest_path.replace("/", "\\")), shell=True)
            subprocess.run("move {} {}".format(infofile_path, dest_path.replace("/", "\\")), shell=True)
        elif env == "lin":
            subprocess.run("mv {} {}".format(datafile_path, dest_path), shell=True)
            subprocess.run("mv {} {}".format(infofile_path, dest_path), shell=True)

    ##write ./"folder_name"/info.txt
    file = open("./" + folder_1 + "info.txt", "w")
    file.write("**{}**\n(train,valid,test)=({},{},{})\n Nu : {}\n Ns : {}\n Nq : {}\n".format(folder_name, train_S, valid_S, test_S, Nu, Ns, Nq))
    file.close()
    finish = time.perf_counter()

    print("\n\nElapsed Time(all) : {}[s]".format(finish-start))
    print("\n  ***    All finished!!!    ***\n")


##main function
if __name__ == "__main__":
    n_proc = -1
    create_target = "dataset"

    args = sys.argv
    for arg in args:
        if "n_proc=" in arg:
            n_proc = int(arg.split("=")[-1])
        elif "one" in args:
            create_target = "one"

    if create_target == "one":
        create_one_data(n_proc)
    elif create_target == "dataset":
        auto_create(n_proc)