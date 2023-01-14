#pragma once

#include <vector>
#include <map>
#include "t-design_util.hpp"
#include <cppsim/state.hpp>

class DataCreator {
private:
    /* member variables */
    //dictionary type of data to store each parameter
    std::map<std::string, std::string> parameters;
    //unitary type(0:Haar, 1:Clifford, 2:LRC, 3:RDC)
    unsigned int unitary_type;
    //size of the teacher dataset
    unsigned int S;
    //number of unitaries contained in one teacher dataset
    unsigned int Nu;
    //number of shots for each unitary
    unsigned int Ns;
    //number of qubits
    unsigned int Nq;
    //circuit depth
    unsigned int depth;
    //seed of psudo random machine
    unsigned int seed;
    //noise type(0:none, 1:Depolarizing, 2:Measurement)
    unsigned int noise_operator;
    //noise ratio(Identity:1-p, noise:p)
    double noise_prob;
    //list of indices when we use in calculating bit correlations
    std::vector<std::vector<int>> comb_list;
    //vec in which measurement probabilities(including bit correlations) are stored
    std::vector<std::vector<float>> teacher_data;
    //list for converting decimal values of sampling results to binary values
    std::vector<std::vector<int>> binary_num_list;
    
    /* member functions */
    //read the configuration file
    std::map<std::string, std::string> _read_inputfile();
    //process to be done before simulation
    void _run_preprocess();
    //functions that actually perform the simulation
    void _haar_sim();
    void _lrc_sim(std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list);
    void _lrc_depolarizing_sim(std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list);
    void _lrc_MeasurementInduced_sim(std::vector<std::vector<std::vector<unsigned int>>>& RU_index_list);
    void _rdc_sim();
    //measurement that guarantees that the value of the sampling result does not exceed 2^n (re-sampling if it does)
    template <typename T> void _state_measurement(T& state, std::vector<ITYPE>& sampling_result);
    //Calculate measurement probabilities from bit strings of sampling results
    std::vector<float> _calc_BitCorr_and_MP(std::vector<ITYPE>& sampling_result);
    //Compute the moment from the results of Nu number of measurement probabilities
    std::vector<float> _calc_moment_of_MP(std::vector<std::vector<float>>& MP_list);

public:
    /* constructor */
    DataCreator() {
        //Set initial parameters
        this->parameters["unitary_type"] = "0";
        this->parameters["S"] = "1000";
        this->parameters["Nu"] = "100";
        this->parameters["Ns"] = "100";
        this->parameters["Nq"] = "4";
        this->parameters["depth"] = "5";
        //this->parameters["seed"] = "8010";
        this->parameters["noise_operator"] = "0";
        this->parameters["noise_prob"] = "0.01";
    }

    /* destructor */
    ~DataCreator() {}

    //set parameters in batch by dictionary type
    void read_configFile();
    //set parameters by key and value
    void set_parameter(std::string key, std::string val);
    //start data generation
    void run_simulation();
    //save simulation results
    void save_result();

};