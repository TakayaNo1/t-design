#include "DataCreator.hpp"

int main(int argc, char *argv[]) {
    //define a class for creating teacher data
    DataCreator Creator;
    //read parameters from config file
    Creator.read_configFile();

    //set parameters from command line arguments
    if(argc > 1) {
        for(int i=1;i<argc;i++) {
            auto parameter_name = split(argv[i], "=").front();
            auto parameter_value = split(argv[i], "=").back();
            Creator.set_parameter(parameter_name, parameter_value);
        }
    }
    //start creating data
    Creator.run_simulation();
    //save the created data
    Creator.save_result();
    
    return 0;
}