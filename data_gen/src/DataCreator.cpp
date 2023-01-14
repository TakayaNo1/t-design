#include "DataCreator.hpp"

#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_general.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <omp.h>


std::map<std::string, std::string> DataCreator::_read_inputfile() {
    //a dictionary type variable which holds parameters of configuration file
    std::map<std::string, std::string> input_paras;

    //file open
    std::ifstream ifs("config_simulation.txt");
    std::string str;

    if (ifs.fail()) {
        //no file
        std::cerr << "Failed to open file. Default setting will be loaded." << std::endl;
    } else {
        //file read
        std::string parameter_name;
        while (getline(ifs, str)) {
            //If the first character is "#", skip it as a comment line.
            if(str[0] != '#') {
                //Split by ":" to get parameter names
                parameter_name = split(str, ":").front();
                auto itr = this->parameters.find(parameter_name);
                //if the key exsits
                if( itr != input_paras.end() ) {
                    input_paras[parameter_name] = split(str, ":").back();
                }
            }
        }
    }
    return input_paras;
}

void DataCreator::read_configFile() {
    //Reflecting the contents of the input file in turn
    auto input_paras = _read_inputfile();
    for(auto itr = input_paras.begin(); itr != input_paras.end(); ++itr) {
        set_parameter(itr->first, itr->second);
    }
}

void DataCreator::set_parameter(std::string key, std::string val) {
    //Update parameter (dictionary type) values based on argument key and value.
    //Update only when the key exists, and an error is made if the key does not exist (do not add a new key).
    if(this->parameters.find(key) != this->parameters.end()) {
        this->parameters[key] = val;
    } else{
        std::cerr << "Key error: " << key << std::endl;
    }
}

template <typename T> void DataCreator::_state_measurement(T& state, std::vector<ITYPE>& sampling_result) {
    //measurement
    sampling_result = state.sampling(this->Ns);
    //the value of the measurement result is 0 to (2^(n-1))
    int max_val = pow(2, this->Nq);
    //Look at the sampling results in turn, and if they are greater than the maximum value, perform re-measurement.
    for(auto&& result : sampling_result) {
        if(result >= max_val) {
            while(true) {
                auto re_sampled = state.sampling(1);
                if(re_sampled[0] < max_val) {
                    std::cout << "Re-sampled: " << result << " => " << re_sampled[0] << std::endl;
                    result = re_sampled[0];
                    break;
                }
            }
        }
    }
}

void DataCreator::_haar_sim() {
    //Vector holding measurement probabilities in each unitary
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //Vector to save measurement results
    std::vector<ITYPE> sampling_result;

    //loop counter
    int i,j;
    
    //creating teacher data
    #pragma omp parallel for private(j, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //state preparation and initialize
            QuantumState state(this->Nq);
            state.set_Haar_random_state();
            //Measurement and calculation of measurement probabilities (and bit correlations)
            sampling_result = state.sampling(this->Ns);
            //_state_measurement(state, sampling_result);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //Calculate moments of measurement probability
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        //Displays master thread progress
        if (omp_get_thread_num() == 0) {
            std::cout << "\r" << i+1 << "/" << this->S/omp_get_num_threads() << " finished...(master thread)" << std::string(20, ' ');
        }
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {

    //Vector holding measurement probabilities in each unitary
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //Vector to save measurement results
    std::vector<ITYPE> sampling_result;

    //loop counter
    int i,j,l;
    //creating teacher data
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //state preparation and initialize
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //perform LRC
            for(l=1;l<this->depth+1;++l) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    auto local_haar_gate = gate::RandomUnitary(qubit_indecies);
                    local_haar_gate->update_quantum_state(&state);
                    delete local_haar_gate;
                }    
            }
            //Measurement and calculation of measurement probabilities (and bit correlations)
            sampling_result = state.sampling(this->Ns);
            //_state_measurement(state, sampling_result);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //Calculate moments of measurement probability
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        //Displays master thread progress
        if (omp_get_thread_num() == 0) {
            std::cout << "\r" << i+1 << "/" << this->S/omp_get_num_threads() << " finished...(master thread)" << std::string(20, ' ');
        }
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_depolarizing_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {
    
    //Vector holding measurement probabilities in each unitary
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //Vector to save measurement results
    std::vector<ITYPE> sampling_result;

    //Define a noise model
    //CPTP that applied each qubit is made as a circuit in advance and apply at each layer.
    QuantumCircuit depolarizing_circuit(this->Nq);
    //create a circuit
    for(int i=0;i<this->Nq;++i) {
        depolarizing_circuit.add_gate(gate::DepolarizingNoise(i, this->noise_prob));
    }

    //loop counter
    int i,j,l;
    //creating teacher data
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //state(density matrix) preparation and initialize
            DensityMatrix state(this->Nq);
            //state.set_zero_state();
            //perform noisy LRCs
            for(l=1;l<this->depth+1;++l) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    auto local_haar_gate = gate::RandomUnitary(qubit_indecies);
                    local_haar_gate->update_quantum_state(&state);
                    delete local_haar_gate;
                }
                //apply noise
                depolarizing_circuit.update_quantum_state(&state);
            }
            //Measurement and calculation of measurement probabilities (and bit correlations)
            sampling_result = state.sampling(this->Ns);
            //_state_measurement(state, sampling_result);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //Calculate moments of measurement probability
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        //Displays master thread progress
        if (omp_get_thread_num() == 0) {
            std::cout << "\r" << i+1 << "/" << this->S/omp_get_num_threads() << " finished...(master thread)" << std::string(20, ' ');
        }
    }
    std::cout << std::endl;
}

void DataCreator::_lrc_MeasurementInduced_sim(
    std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list) {
    
    //Vector holding measurement probabilities in each unitary
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //Vector to save measurement results
    std::vector<ITYPE> sampling_result;

    //Define a random measurement circuit
    //Random measurements that applied each qubit is made as a circuit in advance and apply at each layer.
    QuantumCircuit p_measure_circuit(this->Nq);
    //Identity
    ComplexMatrix dim2_matrix = Eigen::MatrixXd::Zero(2, 2);
    dim2_matrix(0, 0) = 1.0;
    dim2_matrix(1, 1) = 1.0;
    ComplexMatrix kraus_identity = sqrt(1.0-this->noise_prob) * dim2_matrix;
    //0-projective measurement
    dim2_matrix(1, 1) = 0.0;
    ComplexMatrix kraus_measure_0 = sqrt(this->noise_prob) * dim2_matrix;
    //1-projective measurement
    dim2_matrix(0, 0) = 0.0;
    dim2_matrix(1, 1) = 1.0;
    ComplexMatrix kraus_measure_1 = sqrt(this->noise_prob) * dim2_matrix;
    //create a circuit
    for(int i=0;i<this->Nq;++i) {
        p_measure_circuit.add_gate(
            gate::CPTP({
                gate::DenseMatrix(i, kraus_identity),
                gate::DenseMatrix(i, kraus_measure_0),
                gate::DenseMatrix(i, kraus_measure_1)
            })
        );
    }

    //loop counter
    int i,j,l;
    //creating teacher data
    #pragma omp parallel for private(j, l, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //state preparation and initialize
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //perform a LRC
            for(l=1;l<this->depth+1;++l) {
                for(const auto& qubit_indecies : RU_index_list[l%2]) {
                    //insert a random measurement to be applied before 2-qubit local Haar gate
                    for (const auto& qubit_index : qubit_indecies) {
                        auto k1 = gate::DenseMatrix(qubit_index, kraus_identity);
                        auto k2 = gate::DenseMatrix(qubit_index, kraus_measure_0);
                        auto k3 = gate::DenseMatrix(qubit_index, kraus_measure_1);
                        auto prob_measure = gate::CPTP({ k1, k2, k3 });
                        prob_measure->update_quantum_state(&state);
                        delete k1;
                        delete k2;
                        delete k3;
                        delete prob_measure;
                    }
                    auto local_haar_gate = gate::RandomUnitary(qubit_indecies);
                    //Apply 2-qubit local random haar gate
                    local_haar_gate->update_quantum_state(&state);
                    delete local_haar_gate;
                }
                //Do not insert measurements in the last two layers of the LRC
                /*
                if(l<this->depth-1){
                    //apply a random measurement circuit
                    p_measure_circuit.update_quantum_state(&state);
                }
                */
            }
            /*
            //add 2 layer of LRC
            for(l=1;l>-1;--l) {
                //Run with l=1 (odd layer) and l=0 (even layer)
                for(const auto& qubit_indecies : RU_index_list[l]) {
                    gate::RandomUnitary(qubit_indecies)->update_quantum_state(&state);
                }
            }
            */
            //Normalization of quantum states
            state.normalize(state.get_squared_norm());
            //Measurement and calculation of measurement probabilities (and bit correlations)
            sampling_result = state.sampling(this->Ns);
            //_state_measurement(state, sampling_result);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //Calculate moments of measurement probability
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        //Displays master thread progress
        if (omp_get_thread_num() == 0) {
            std::cout << "\r" << i+1 << "/" << this->S/omp_get_num_threads() << " finished...(master thread)" << std::string(20, ' ');
        }
    }
    std::cout << std::endl;
}

void DataCreator::_rdc_sim() {
    //Vector holding measurement probabilities in each unitary
    //std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
    //Vector to save measurement results
    std::vector<ITYPE> sampling_result;
    
    //Circuit to apply Hadmard to all qubits, changing the basis Z <=> X.
    QuantumCircuit basis_change_circuit(this->Nq);
    for(int i=0;i<this->Nq;++i) {
        basis_change_circuit.add_H_gate(i);
    }

    //loop counter
    int i,j,d;
    //creating teacher data
    #pragma omp parallel for private(j, d, sampling_result)
    for(i=0;i<this->S;++i) {
        std::vector<std::vector<float>> MP_list(this->Nu, std::vector<float>(this->comb_list.size()));
        for(j=0;j<this->Nu;++j) {
            //state preparation and initialize
            QuantumState state(this->Nq);
            //state.set_zero_state();
            //perform RDCs
            for(d=0;d<this->depth;++d) {
                state.multiply_elementwise_function(gen_RandomDiagonal_element);
                basis_change_circuit.update_quantum_state(&state);
            }
            //Measurement and calculation of measurement probabilities (and bit correlations)
            sampling_result = state.sampling(this->Ns);
            //_state_measurement(state, sampling_result);
            MP_list[j] = _calc_BitCorr_and_MP(sampling_result);
        }
        //Calculate moments of measurement probability
        this->teacher_data[i] = _calc_moment_of_MP(MP_list);
        //Displays master thread progress
        if (omp_get_thread_num() == 0) {
            std::cout << "\r" << i+1 << "/" << this->S/omp_get_num_threads() << " finished...(master thread)" << std::string(20, ' ');
        }
    }
    std::cout << std::endl;
}

void DataCreator::_run_preprocess() {
    //Take a value and set it to each variable
    this->unitary_type = std::stoi(this->parameters["unitary_type"]);
    this->S = std::stoi(this->parameters["S"]);
    this->Nu = std::stoi(this->parameters["Nu"]);
    this->Ns = std::stoi(this->parameters["Ns"]);
    this->Nq = std::stoi(this->parameters["Nq"]);
    this->depth = std::stoi(this->parameters["depth"]);
    //this->seed = std::stoi(this->parameters["seed"]);    
    this->noise_operator = std::stoi(this->parameters["noise_operator"]);
    this->noise_prob = std::stod(this->parameters["noise_prob"]);
    
    //output the config
    std::cout << "** Parameters **" << std::endl;
    for(auto itr = this->parameters.begin(); itr != this->parameters.end(); ++itr) {
        std::cout << itr->first << " : " << itr->second << std::endl;
    }
    std::cout << std::endl;

    //Get a list of the locations of the qubits for which bit correlations are to be computed
    //Pre-calculate all possible bit correlations from point 1 to Nq
    this->comb_list.clear();
    this->comb_list = get_possibliyMax_bitCorr(this->Nq);

    //Pre-determine the size of the vector in which measurement probabilities are stored
    this->teacher_data.clear();
    this->teacher_data.resize(this->S, std::vector<float>(this->comb_list.size()));

    //Create a list of binary numbers
    this->binary_num_list.clear();
    int num_decimal;
    for (int i = 0; i < pow(2, this->Nq); ++i) {
        //Decimal value
        num_decimal = i;
        //The value converted to binary and stored in vector.
        //Initialize with -1 and rewrite only the bits where 1 stands, and the conversion will be 0=>-1.
        std::vector<int> bit_Nq_size(this->Nq, -1);
        if (num_decimal != 0) {
            for (int j = 0; j < log2(i) + 1; ++j) {
                if (num_decimal % 2 == 1) {
                    bit_Nq_size[this->Nq - 1 - j] = 1;
                }
                num_decimal /= 2;
            }
        }
        this->binary_num_list.emplace_back(bit_Nq_size);
    }
}

void DataCreator::run_simulation() {
    //obtain the current time
    auto start = std::chrono::system_clock::now();

    //run preprocess
    _run_preprocess();

    if(this->unitary_type == 0) {
        _haar_sim();
    }
    else if(this->unitary_type == 1) {
        std::cerr << "**Random Clifford is selected, but not implemented yet." << std::endl;
        std::cerr << "**Please use python code(generated data is 0)." << std::endl;
    }
    else if(this->unitary_type == 2) {
        /**
         * Prepare a list of indices that will be applied to 2-qubit Haar random unitary.
         * Passed in circuit.add_Random_unitary_gate().
         * The structure is a 3d vector, where the first dim is even or odd depth and
         * the second dim is index of a 2-qubit Haar gate, and the third dim is the index of the qubit.
         * (例) if 6-qubit LRC, 
         *      RU_index_list = [[[1,2],   [3,4]          ],  <= if depth is even([0])
         *                        [0,1],   [2,3],   [4,5] ]]  <= if depth is odd([1])
         *                        ↑Index of the qubit that the first 2-qubit Haar will be applied
         */
        //3d vector holding the index of the qubits over which the 2-qubit Haar gate
        //is applied when the depth is odd and even
        std::vector<std::vector<std::vector<unsigned int>>> RU_index_list;
        //Vector that holds either index when depth is odd or even
        std::vector<std::vector<unsigned int>> target_index;
        //1d vector holding the index of a qubit of length 2
        std::vector<unsigned int> index_twoQubit;
        //begin creation
        int count = 0;
        //create a list of the indexes of the qubits when the depth is odd
        for(int i=1;i<Nq;++i) {
            index_twoQubit.emplace_back(i);
            count++;
            if(count == 2){
                target_index.emplace_back(index_twoQubit);
                index_twoQubit.clear();
                count = 0;
            }
        }
        RU_index_list.emplace_back(target_index);
        target_index.clear();
        index_twoQubit.clear();
        count = 0;
        //create a list of the indexes of the qubits when the depth is even
        unsigned int maxIndex_depthOdd;
        if (Nq%2 == 1) {
            maxIndex_depthOdd = Nq - 1;
        } else {
            maxIndex_depthOdd = Nq;
        }
        for(int i=0;i<maxIndex_depthOdd;++i) {
            index_twoQubit.emplace_back(i);
            count++;
            if(count == 2){
                target_index.emplace_back(index_twoQubit);
                index_twoQubit.clear();
                count = 0;
            }
        }
        RU_index_list.emplace_back(target_index);

        if(this->noise_operator == 0) {
            _lrc_sim(RU_index_list);
        }
        else if(this->noise_operator == 1) {
            _lrc_depolarizing_sim(RU_index_list);
        }
        else if(this->noise_operator == 2) {
            _lrc_MeasurementInduced_sim(RU_index_list);
        }
    }
    else if(this->unitary_type == 3) {
        _rdc_sim();
    }
    else {
        std::cerr << "**Simulation will not start. Check the value of unitary_type." << std::endl;
    }

    //obtain the current time
    auto finish = std::chrono::system_clock::now();
    //Convert the time which required for processing to milliseconds and divide by 1000, then get execution time[s]
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() / 1000.0;
    std::cout << "Creating time : " << elapsed << "[s]" << std::endl;
}

std::vector<float> DataCreator::_calc_BitCorr_and_MP(std::vector<ITYPE>& sampling_result) {
    //List of measurement probabilities
    std::vector<float> MP_list;

    //Bit correlations are calculated for each measurement result, so bit correlation values for shots are obtained.
    //Add them up and finally divide by the number of measurements to obtain the measurement probability
    //(expected value =: bit correlation value).
    float sum_bitcorr;
    int bitcorr_oneshot;

    for (const auto& bit_index_list : this->comb_list) {
        //Get position to calculate bit correlations
        sum_bitcorr = 0.0;
        for (const auto& result : sampling_result) {
            bitcorr_oneshot = 1;
            for (const auto& qubit_index : bit_index_list) {
                //bitcorr_oneshot *= this->binary_num_list[result][qubit_index];
                bitcorr_oneshot *= this->binary_num_list.at(result).at(qubit_index);
            }
            sum_bitcorr += bitcorr_oneshot;
        }
        MP_list.emplace_back(sum_bitcorr / this->Ns);
    }

    return MP_list;
}

std::vector<float> DataCreator::_calc_moment_of_MP(std::vector<std::vector<float>>& MP_data) {
    int Nq_prime = MP_data[0].size();

    std::vector<float> moments_of_MP;
    moments_of_MP.reserve(Nq_prime*20);

    std::vector<float> MP_mom_eachDim(Nq_prime, 0.0);

    for(int dim=1;dim<21;++dim) {
        for(const auto& MP_eachU : MP_data) {
            for(int j=0;j<Nq_prime;++j) {
                MP_mom_eachDim[j] += pow(MP_eachU[j], dim);
                //MP_mom_eachDim[j] += fabs(pow(MP_eachU[j], dim));
            }
        }
        for(auto&& each_MP_mom : MP_mom_eachDim) {
            moments_of_MP.emplace_back(each_MP_mom / this->Nu);
            each_MP_mom = 0.0;
        }
    }
    
    return moments_of_MP;
}

void DataCreator::save_result() {
    //Obtain ID (time)
    std::string date_ID = getDatetimeStr();
    std::string unitary_str;
    //set the operator
    if (this->unitary_type == 0) {
        unitary_str = "haar";
    } else if (this->unitary_type == 1) {
        unitary_str = "clif";
    } else if (this->unitary_type == 2) {
        unitary_str = "lrc";
    } else if (this->unitary_type == 3) {
        unitary_str = "rdc";
    }

    //Save measurement probabilities as CSV files
    std::string file_name = "../result/" + unitary_str + "_" + date_ID + ".csv";
    std::ofstream csv_file(file_name);
    for(const auto& each_S : this->teacher_data) {
        for(auto itr=each_S.begin(); itr!=each_S.end()-1; ++itr) {
            csv_file << *itr << ",";
        }
        csv_file << each_S.back() << "\n";
    }
    csv_file.close();

    //Save parameters and other information during simulation to the text file
    file_name = "../result/info_" + unitary_str + '_' + date_ID + ".txt";
    std::ofstream config_file(file_name);
    config_file << "|S| : " << this->S << std::endl;
    config_file << " Nu : " << this->Nu << std::endl;
    config_file << " Ns : " << this->Ns << std::endl;
    config_file << " Nq : " << this->Nq << std::endl;
    if(this->unitary_type == 2 || this->unitary_type == 3) {
        config_file << " depth : " << this->depth << std::endl;
    }
    //config_file << " seed : " << this->seed << std::endl;
    if(this->unitary_type == 2) {
        if(this->noise_operator == 0) {
            config_file << " noise_operator : none" << std::endl;
        } else if(this->noise_operator == 1) {
            config_file << " noise_operator : DepolarizingNoise" << std::endl;
            config_file << " noise_prob : " << this->noise_prob << std::endl;
        } else if(this->noise_operator == 2) {
            config_file << " noise_operator : Measurement-Induced" << std::endl;
            config_file << " noise_prob : " << this->noise_prob << std::endl;
        }
    }
    config_file << " bit corrlation : " << this->Nq << std::endl;
    config_file << " dim of moments : 1~20 " << std::endl;
    config_file.close();
}