#define _USE_MATH_DEFINES
#include <cppsim/state.hpp>
#include <Eigen/QR>
#include <Eigen/LU>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <deque>
#include <chrono>
#include <iomanip>
#include <string>
#include <complex>
//#include <Eigen/KroneckerProduct>
#include <experimental/filesystem>
#include <cmath>
#include <chrono>

#include <boost/multiprecision/cpp_int.hpp>
namespace mp = boost::multiprecision;

ComplexMatrix my_tensor_product(ComplexMatrix A, ComplexMatrix B) {
    /** Function to compute the tensor product (Kronecker product, np.kron(A,B))
     *   Args:
     *       A,B(ComplexMatrix) := Complex matrices
     *   Return:
     *       A tensor B (np.kron(A,B)) := Matrix of the result of the tensor product of A and B
     */
    //Get the dimension of A and B matrices
    unsigned long long int mat_A_dim = A.rows();
    unsigned long long int mat_B_dim = B.rows();
    //Declare a matrix to save the calculation results
    ComplexMatrix C(mat_A_dim*mat_B_dim, mat_A_dim*mat_B_dim);
    //Declare a matrix that represents some blocks of the matrix
    ComplexMatrix small_mat;

    //Calculation of tensor products
    for (int i = 0; i < mat_A_dim; i++) {
        for (int j = 0; j < mat_A_dim; j++) {
            //Calculate the scalar multiples of matrix B by the elements of matrix A to create a block matrix
            small_mat = A(i, j) * B;
            //Insert the block matrix value to the appropriate part of the matrix after the tensor product
            for (int k = 0; k < mat_B_dim; k++) {
                for (int l = 0; l < mat_B_dim; l++) {
                    C(k + i * mat_B_dim, l + j * mat_B_dim) = small_mat(k, l);
                }
            }
        }
    }
    return C;
}

/**
 * Ref: https://github.com/qulacs/qulacs/blob/master/src/cppsim/gate_factory.cpp#L180
 */
ComplexMatrix gen_haar_RU(unsigned int num_qubits) {
    /** Function to generate an n-qubit Haar Random Unitary.
     *  Implemented with reference to the one implemented in Qulacs.
     *  Arg:
     *      num_qubit(int) := number of qubit
     *  Return:
     *      Q(ComplexMatrix) := matrix of Haar Random Unitary
     */
    Random random;
    unsigned long long int dim = pow(2, num_qubits);
    ComplexMatrix matrix(dim, dim);
    for (unsigned long long int i = 0; i < dim; ++i) {
        for (unsigned long long int j = 0; j < dim; ++j) {
            matrix(i, j) = (random.normal() + 1.i * random.normal()) / sqrt(2.);
        }
    }
    Eigen::HouseholderQR<ComplexMatrix> qr_solver(matrix);
    ComplexMatrix Q = qr_solver.householderQ();
    // actual R matrix is upper-right triangle of matrixQR
    auto R = qr_solver.matrixQR();
    for (unsigned long long int i = 0; i < dim; ++i) {
        CPPCTYPE phase = R(i, i) / abs(R(i, i));
        for (unsigned long long int j = 0; j < dim; ++j) {
            Q(j, i) *= phase;
        }
    }
    return Q;
}

/** Function to return the current date and time as string
 * Used to specify initial seed values, file names, etc.
 * Example) 22:58:45, June 29, 2021  => "20210629225845"
 */
inline std::string getDatetimeStr() {
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
    //Linux(GNU, intel Compiler)
    time_t t = time(nullptr);
    const tm* localTime = localtime(&t);
    std::stringstream s;
    s << "20" << localTime->tm_year - 100;
    //zerofill using setw() and setfill()
    s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime->tm_min;
    s << std::setw(2) << std::setfill('0') << localTime->tm_sec;
#elif _MSC_VER
    //Windows(Visual C++ Compiler)
    time_t t;
    struct tm localTime;
    time(&t);
    localtime_s(&localTime, &t);
    std::stringstream s;
    s << "20" << localTime.tm_year - 100;
    //zerofill using setw() and setfill()
    s << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime.tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime.tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime.tm_min;
    s << std::setw(2) << std::setfill('0') << localTime.tm_sec;
#endif
    //return the value as std::string
    return s.str();
}

enum CulculateType{
    /**
     * U = {U_1 .. U_M}
     * F_2 = M^-1 * (M-1)^-1 * Σ[i!=j]|Tr(U_i * dag(U_j))|^(2t)
    */
    B12 = 0,
    /**
     * U = {U_1 .. U_M}
     * U[i] = {U_1 .. U_i-1, U_i+1, .. U_M}
     * ps[i] = M * F_2(U) - (M - 1) * F_2(U[i])
     * ps = n^-1 * Σ ps[i]
     * Vps = (n-1)^-1 * Σ (ps[i]-ps)^2
     * jackknife confidence interval : ps +- 0.960 * sqrt(Vps / M)
    */
    B12_Jackknife = 1,
    /**
     * U = {U_1 .. U_M}
     * V = {U_1 .. U_M}
     * F_3 = M^-1 * Σ|Tr(U_i * dag(V_i))|^(2t)
    */
    B15 = 2
};

class CliffordGroup{
private:
    int Nq;
    mp::cpp_int order=1;
    std::vector<ComplexMatrix> A,B,C,D,E;
    std::vector<mp::cpp_int> Lcounts = {0};
    std::vector<mp::cpp_int> Rcounts = {0};
    std::vector<mp::cpp_int> Lsum = {0};

    unsigned long long pow2(int n){
        unsigned long long r=1;
        for(int i=0;i<n;i++)r*=2;
        return r;
    }
    ComplexMatrix identity(long n){
        return Eigen::MatrixXd::Identity(n, n);
    }
    ComplexMatrix pickM(int n, mp::cpp_int index){
        ComplexMatrix d, e, mat;
        mat=identity(pow2(n));
        
        for(int i=0;i<n-1;i++){
            d=D[(int)(index%D.size())];
            index=index/D.size();
            d=my_tensor_product(identity(pow2(i)), d);
            d=my_tensor_product(d, identity(pow2(n-2-i)));
            mat=mat*d;
        }

        e=E[(int)(index%E.size())];
        e=my_tensor_product(identity(pow2(n-1)), e);
        mat=mat*e;
        return mat;
    }
    ComplexMatrix pickLm(int n,int m, mp::cpp_int index){
        assert(m<=n);

        //ComplexMatrix mat = Eigen::MatrixXd::Identity(N,N);
        ComplexMatrix a, b, c, mat;
        a = A[(int)(index%A.size())];
        index = index/A.size();
        a=my_tensor_product(identity(pow2(m-1)), a);
        mat=my_tensor_product(a, identity(pow2(n-m)));

        for(int i=0;i<m-1;i++){
            b=B[(int)(index%B.size())];
            index=index/B.size();
            b=my_tensor_product(identity(pow2(m-2-i)),b);
            b=my_tensor_product(b,identity(pow2(n-m+i)));
            mat=mat*b;
        }

        c=C[(int)(index%C.size())];
        c=my_tensor_product(c,identity(pow2(n-1)));
        mat=mat*c;
        return mat;
    }
    ComplexMatrix pickL(int n, mp::cpp_int index){
        assert(index < Lsum.back());

        ComplexMatrix mat;
        for(int m=1;m<=n;m++){
            if(index < Lsum[m]){
                mat=pickLm(n,m,index-Lsum[m-1]);
                break;
            }
        }
        return mat;
    }

public:
    CliffordGroup(){}
    CliffordGroup(int Nq){
        this->Nq=Nq;

        ComplexMatrix I,X,Y,Z,H,S,CZ,IH,HI,HH,SI,IS,A1,A2,A3,B1,B2,B3,B4,C1,C2,D1,D2,D3,D4,E1,E2,E3,E4;
        I = Eigen::MatrixXd::Identity(2, 2);
        X = Eigen::MatrixXd::Zero(2,2);
        X << 0,1,1,0;
        Y = Eigen::MatrixXd::Zero(2,2);
        Y << 0,-1i,1i,0;
        Z = Eigen::MatrixXd::Zero(2,2);
        Z << 1,0,0,-1;
        H = Eigen::MatrixXd::Zero(2,2);
        H(0, 0) = (1/sqrt(2)) * 1;
        H(0, 1) = (1/sqrt(2)) * 1;
        H(1, 0) = (1/sqrt(2)) * 1;
        H(1, 1) = (1/sqrt(2)) * -1;
        S = Eigen::MatrixXd::Zero(2,2);
        S << 1,0,0,1i;
        CZ = Eigen::MatrixXd::Zero(4,4);
        CZ << 1,0,0,0,
              0,1,0,0,
              0,0,1,0,
              0,0,0,-1;
        IH = my_tensor_product(I,H);
        HI = my_tensor_product(H,I);
        HH = my_tensor_product(H,H);
        SI = my_tensor_product(S,I);
        IS = my_tensor_product(I,S);
        A1 = I;
        A2 = H;
        A3 = H*S*H;
        B1 = IH*CZ*HH*CZ*HH*CZ;
        B2 = CZ*HH*CZ;
        B3 = HI*SI*CZ*HH*CZ;
        B4 = HI*CZ*HH*CZ;
        C1 = I;
        C2 = H*S*S*H;
        D1 = CZ*HH*CZ*HH*CZ*IH;
        D2 = HI*CZ*HH*CZ*IH;
        D3 = HH*IS*CZ*HH*CZ*IH;
        D4 = HH*CZ*HH*CZ*IH;
        E1 = I;
        E2 = S;
        E3 = S*S;
        E4 = S*S*S;
        A = {A1,A2,A3};
        B = {B1,B2,B3,B4};
        C = {C1,C2};
        D = {D1,D2,D3,D4};
        E = {E1,E2,E3,E4};

        for(int i=0;i<Nq;i++){
            if(i==0) Lcounts.push_back(C.size()*A.size());
            else Lcounts.push_back(Lcounts.back()*B.size());
        }
        for(int i=0;i<Nq;i++){
            if(i==0) Rcounts.push_back(E.size());
            else Rcounts.push_back(Rcounts.back()*D.size());
        }
        for(int i=0;i<Nq;i++){
            Lsum.push_back(Lsum.back()+Lcounts[i+1]);
        }
        order=1;
        for(int i=1;i<=Nq;i++){
            order *= 2 * (pow2(2*i)-1) * pow2(2*i);
        }
    }

    ComplexMatrix getElement(mp::cpp_int index){
        assert(0 <= index && index < order);

        ComplexMatrix l, m, mat;
        mp::cpp_int count=1;
        mat=identity(pow2(Nq));

        for(int n=Nq;n>0;n--){
            l=pickL(n, index%Lsum[n]);
            index=index/Lsum[n];
            count*=Lsum[n];
            mat=mat*my_tensor_product(l, identity(pow2(Nq-n)));

            m=pickM(n, index%Rcounts[n]);
            index=index/Rcounts[n];
            count*=Rcounts[n];
            mat=mat*my_tensor_product(m, identity(pow2(Nq-n)));
        }
        assert(count==order);

        mat /= pow(mat.determinant(), pow(2, -Nq));
        return mat;
    }

    ComplexMatrix sample(){
        Random random;
        // mp::cpp_int r1 = random.int64();
        // mp::cpp_int r2 = random.int64();
        // mp::cpp_int random_index = (r1<<64 | r2) % order;
        
        mp::cpp_int random_index = 0;
        mp::cpp_int tmp=order, r;
        while(tmp>0){
            r = random.int64();
            random_index = (random_index<<64) | r;
            tmp=tmp>>64;
        }
        r = random.int64();
        random_index = (random_index<<64) | r;
        
        
        // std::cout<<random_index << ", "<<order <<std::endl;
        random_index = random_index % order;
        // std::cout<<random_index << ", "<<order <<std::endl;
        
        return getElement(random_index);
    }
};

class FramePotential {
private:
    /* member variable */
    unsigned int num_qubits;    //number of qubit
    unsigned int depth;         //the circuit depth
    unsigned long long int dim; //dimention of a system
    unsigned int t;             //the order of t of t-design
    unsigned int num_sample;    //number of unitary samples
    double epsilon;             //convergence error
    unsigned int patience;      //Number used to determine convergence
   
    std::string circuit;        //Specify "LRC" or "RDC" as string type
    unsigned int num_gates_depthOdd;  //Number of 2-qubit Haar random unitaries at odd order depths
    unsigned int num_gates_depthEven; //Number of 2-qubit Haar random unitaries at even order depths

    enum CulculateType culcType; 

    std::vector<double> result_oneshot;
    std::vector<double> result_mean;

    /* sample unitary randomly */
    ComplexMatrix sample_unitary();

    /* determine convergence */
    bool check_convergence();

public:
    /* constructor */
    FramePotential(std::string circ) {
        this->circuit = circ;
        this->num_qubits = 3;
        this->depth = 3;
        this->dim = 8;
        this->t = 3;
        this->num_gates_depthOdd = 1;
        this->num_gates_depthEven = 1;

        this->culcType = CulculateType::B12;

        this->result_oneshot.clear();
        this->result_mean.clear();
    }

    CliffordGroup cliffordGroup;

    /* setter of parameters */
    void set_paras(unsigned int Nq, unsigned int D, unsigned int dim_t, unsigned int Nsample, double eps, unsigned int pat);
    void set_culculateType(enum CulculateType type){
        this->culcType=type;
    }

    /* calculate the value of Frame Potential until it converges */
    void calculate();
    void clear_result();

    /* output the calculation result */
    void output();

    /* File output of calculation results */
    void save_result(std::string file_name);

    /* getter of calculation results */
    double get_result();
};

void FramePotential::set_paras(unsigned int Nq, unsigned int D, unsigned int dim_t, unsigned int Nsample, double eps = 0.0001, unsigned int pat = 5) {
    //Parameter setting from arguments
    this->num_qubits = Nq;
    this->dim = pow(2, Nq);
    this->depth = D;
    this->t = dim_t;
    this->num_sample = Nsample;
    this->epsilon = eps;
    this->patience = pat;

    //Preparation of "LRC"
    if(this->circuit == "RC"){
        this->cliffordGroup = CliffordGroup(Nq);
    } else if (this->circuit == "LRC") {
        this->num_gates_depthOdd = Nq / 2;
        if (Nq % 2 == 1) {
            this->num_gates_depthEven = Nq / 2;
        }
        else {
            this->num_gates_depthEven = (Nq / 2) - 1;
        }
    //Assign 0 for the case other than "LRC"
    } else {
        this->num_gates_depthOdd = 0;
        this->num_gates_depthEven = 0;
    }
}

void FramePotential::calculate() {
    unsigned long long int count = 0;
    ComplexMatrix U, Vdag;
    
    if(this->culcType == CulculateType::B12){
        /*B12*/
        double total=0, result;
        std::vector<ComplexMatrix> samples;

        // auto t1 = std::chrono::system_clock::now();
        for(int i=0;i<this->num_sample;i++){
            samples.push_back(sample_unitary());
        }
        // auto t2 = std::chrono::system_clock::now();
        // auto dur = t2 - t1;
        // auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        // std::cout << "sample time : " << msec << " milli sec." << std::endl;
        
        for(int i=0;i<this->num_sample;i++){
            for(int j=i+1;j<this->num_sample;j++){
                count++;
                U=samples[i];
                Vdag=samples[j].adjoint();
                result=pow(abs((Vdag*U).trace()), 2. * this->t);
                total+=result;
            }
        }
        // auto t3 = std::chrono::system_clock::now();
        // dur = t3 - t2;
        // msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        // std::cout << "calculate time : " << msec << " milli sec." << std::endl;
        std::cout << total/count << std::endl;
    }else if(this->culcType == CulculateType::B12_Jackknife){
        double total=0, result, total_n_th_value_deleted;
        double *results=new double[this->num_sample*(this->num_sample-1)/2];
        std::vector<ComplexMatrix> samples;
        double fp_sample;
        double *fp_sample_n_th_value_deleted=new double[this->num_sample];
        double *psi=new double[this->num_sample];
        int i,j,n;

        // auto t1 = std::chrono::system_clock::now();
        for(i=0;i<this->num_sample;i++){
            samples.push_back(sample_unitary());
        }

        // auto t2 = std::chrono::system_clock::now();
        // auto dur = t2 - t1;
        // auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        // std::cout << "sample time : " << msec << " milli sec." << std::endl;
        
        for(i=0;i<this->num_sample;i++){
            for(j=i+1;j<this->num_sample;j++){
                count++;
                U=samples[i];
                Vdag=samples[j].adjoint();
                result=pow(abs((Vdag*U).trace()), 2. * this->t);
                results[i*this->num_sample-i*(i+1)/2+(j-i-1)] = result;
                total+=result;
            }
        }
        samples.clear();
        fp_sample = total/count;

        for(n=0;n<this->num_sample;n++){
            // total=0;
            // count=0;

            // for(i=0;i<this->num_sample;i++){
            //     for(j=i+1;j<this->num_sample;j++){
            //         if(i == n || j == n)continue;

            //         count++;
            //         total+=results[i*this->num_sample-i*(i+1)/2+(j-i-1)];
            //     }
            // }
            // fp_sample_n_th_value_deleted[n]=total/count;
            total_n_th_value_deleted=total;
            for(i=0;i<n;i++){
                total_n_th_value_deleted-=results[i*this->num_sample-i*(i+1)/2+(n-i-1)];
            }
            for(j=n+1;j<this->num_sample;j++){
                total_n_th_value_deleted-=results[n*this->num_sample-n*(n+1)/2+(j-n-1)];
            }
            fp_sample_n_th_value_deleted[n]=total_n_th_value_deleted/(count-this->num_sample+1);
        }
        for(int i=0;i<this->num_sample;i++){
            psi[i] = this->num_sample * fp_sample - (this->num_sample-1) * fp_sample_n_th_value_deleted[i];
        }
        // auto t3 = std::chrono::system_clock::now();
        // dur = t3 - t2;
        // msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        // std::cout << "calculate time : " << msec << " milli sec." << std::endl;
        std::cout << "value" << std::endl;
        for(int i=0;i<this->num_sample;i++){
            std::cout << psi[i] << std::endl;
        }

        delete results, fp_sample_n_th_value_deleted, psi;
    }else if(this->culcType == CulculateType::B15){
        /*B15*/
        while (true) {
            count++;
            U = sample_unitary();
            Vdag = sample_unitary().adjoint();
            this->result_oneshot.emplace_back(pow(abs((U*Vdag).trace()), 2. * this->t));
            std::cout << "\r" << "  calculated " << count << " times..." << std::string(10, ' ');
            if (check_convergence()) {
                break;
            }
        }
        std::cout << std::endl;
    }
}

void FramePotential::clear_result(){
    this->result_oneshot.clear();
    this->result_mean.clear();
}

ComplexMatrix FramePotential::sample_unitary() {
    //the unitary matrix that will eventually be returned
    ComplexMatrix big_unitary = Eigen::MatrixXd::Identity(this->dim, this->dim);

    if (this->circuit == "RC"){
        big_unitary=this->cliffordGroup.sample();
    }else if (this->circuit == "LRC") {
        //Unitary matrix of one layer of LRC
        ComplexMatrix small_unitary;
        for (int i = this->depth; i > 0; i--) {
            //generate 2-qubit Haar random unitary and take tensor product
            if (i % 2 == 1) {
                //if the depth is odd
                small_unitary = gen_haar_RU(2);
                for (int j = 0; j < this->num_gates_depthOdd - 1; j++) {
                    small_unitary = my_tensor_product(small_unitary, gen_haar_RU(2));
                }
                //When both the number of qubits and the depth of the circuit are odd, the Identity is added at the end.
                if (this->num_qubits % 2 == 1) {
                    small_unitary = my_tensor_product(small_unitary, Eigen::MatrixXd::Identity(2, 2));
                }
            }
            else {
                //Whenever the circuit depth is even, the first is always Identity
                small_unitary = Eigen::MatrixXd::Identity(2, 2);
                for (int j = 0; j < this->num_gates_depthEven; j++) {
                    small_unitary = my_tensor_product(small_unitary, gen_haar_RU(2));
                }
                //If the number of qubits is even and the depth of the circuit is even, put Identity at the end.
                if (this->num_qubits % 2 == 0) {
                    small_unitary = my_tensor_product(small_unitary, Eigen::MatrixXd::Identity(2, 2));
                }
            }
            //Merging "num_qubits" size of unitaries created for each depth by applying
            big_unitary *= small_unitary;
        }
    }
    else if(this->circuit == "RDC") {
        //create 1-qubit Hadamard matrix
        ComplexMatrix hadmard_matrix_2d = Eigen::MatrixXd(2, 2);
        hadmard_matrix_2d(0, 0) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(0, 1) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(1, 0) = (1/sqrt(2)) * 1;
        hadmard_matrix_2d(1, 1) = (1/sqrt(2)) * -1;
        //create "num_qubits"-qubit Hadamard matrix
        ComplexMatrix hadmard_matrix = hadmard_matrix_2d;
        for(int i=0;i<this->num_qubits-1;i++) {
            hadmard_matrix = my_tensor_product(hadmard_matrix, hadmard_matrix_2d);
        }
        //Create a vector whose elements are the diagonal components of a Random Diagonal Unitary
        ComplexMatrix RDU = Eigen::MatrixXd::Identity(this->dim, this->dim);
        //create a RDC
        for(int i=0;i<this->depth;i++) {
            //Put values in the diagonal components of RDU
            for(int j=0;j<this->dim;j++) {
                Random random;
                RDU(j,j) = std::exp(std::complex<float>(0.0, random.uniform() * 2 * M_PI));
            }
            //Repeat alternately applying RDU and "num_qubits"-qubit Hadmard for depth times
            big_unitary *= hadmard_matrix * RDU;
        }
    }
    else {
        std::cerr << "CAUTION: The circuit '" << this->circuit << "' is not implemented. Identity will be returned." << std::endl;
    }

    return big_unitary;
}

bool FramePotential::check_convergence() {
    bool flag = true;
    unsigned long long int num_calclated = this->result_oneshot.size();

    if (num_calclated == 1) {
        this->result_mean.emplace_back(result_oneshot[0]);
        flag = false;
    }
    else if (num_calclated < this->patience) {
        this->result_mean.emplace_back(((num_calclated - 1)*this->result_mean.back() + result_oneshot.back()) / num_calclated);
        flag = false;
    }
    else {
        this->result_mean.emplace_back(((num_calclated - 1)*this->result_mean.back() + result_oneshot.back()) / num_calclated);
        for (int i = 0; i < this->patience; i++) {
            if (abs(this->result_mean.back() - this->result_mean[num_calclated - 1 - this->patience + i]) >= this->epsilon) {
                flag = false;
                break;
            }
        }
    }
    return flag;
}

void FramePotential::output() {
    std::cout << std::endl << "*** Result ***" << std::endl;
    std::cout << "num_qubits : " << this->num_qubits << std::endl;
    std::cout << "depth : " << this->depth << std::endl;
    std::cout << "t : " << this->t << std::endl;
    std::cout << "epsilon : " << this->epsilon << std::endl;
    std::cout << "patience : " << this->patience << std::endl;
    std::cout << "FramePotential : " << this->result_mean.back() << std::endl << std::endl;
}

void FramePotential::save_result(std::string file_name = getDatetimeStr() + ".csv") {
    std::string path = "./result/" + file_name;

    std::ofstream csv_file(path);
    csv_file << "circ:" << this->circuit << ",t:" << this->t << std::endl;
    csv_file << "Nq:" << this->num_qubits << ",depth:" << this->depth << std::endl;
    csv_file << "epsilon:" << this->epsilon << ",patience:" << this->patience << std::endl;
    csv_file << "shot,average" << std::endl;

    for (unsigned long long int i = 0; i < this->result_oneshot.size(); i++) {
        csv_file << this->result_oneshot[i] << "," << this->result_mean[i] << std::endl;
    }
    csv_file.close();
}

double FramePotential::get_result() {
    return this->result_mean.back();
}


int test() {
    //Execution example
    /* set parameters */
    int ntimes = 10;
    std::vector<int> Nq_list = { 7 };
    std::vector<int> depth_list = { 11,12,13,14,15 };
    std::vector<int> t_list = { 2,3,4,5 };
    std::vector<int> Nsample_list = {100};
    double eps = 0.0001;
    unsigned int pat = 5;
    /* Specify the circuit */
    std::string circ_type = "LRC";
    //std::string circ_type = "RDC";
    
    /* call the class */
    FramePotential FP = { circ_type };

    /* begin calculation */
    for(int l = 0; l < Nsample_list.size(); l++){
        for (int i = 0; i < t_list.size(); i++) {
            for (int j = 0; j < Nq_list.size(); j++) {
                for (int k = 0; k < depth_list.size(); k++) {
                    //set parameters
                    std::cout << "sample:" << Nsample_list[l] << ", t:" << t_list[i] << ", depth:" << depth_list[k] << std::endl;
                    FP.set_paras(Nq_list[j], depth_list[k], t_list[i], Nsample_list[l], eps, pat);
                    //Variables for calculating mean and standard deviation values
                    double ave = 0.0, std = 0.0;
                    //repeat "n" times
                    for(int n=0; n < ntimes; n++) {
                        FP.clear_result();
                        //begin calculating the value of Frame Potential
                        FP.calculate();
                        //get the calculation results and add them up
                        ave += FP.get_result();
                        std += (FP.get_result() * FP.get_result());
                    }
                    //calculate the average
                    ave /= ntimes;
                    //calculate the standard deviation
                    std = sqrt(std / ntimes - ave * ave);

                    std::cout << "\tave:" << ave << ", std:" << std << std::endl;
                }
            }
        }
    }

    /*
    for (int i = 0; i < t_list.size(); i++) {
        for (int j = 0; j < Nq_list.size(); j++) {
            for (int k = 0; k < depth_list.size(); k++) {
                //set parameters
                FP.set_paras(Nq_list[j], depth_list[k], t_list[i], eps, pat);
                std::cout << std::endl << "Now => Nq:" << Nq_list[j] << ", depth:" << depth_list[k] << ", t:" << t_list[i] << std::endl;
                //set directory name
                std::string dir_name = "B12_"+circ_type
                                        + "_Nq" + std::to_string(Nq_list[j])
                                        + "_depth" + std::to_string(depth_list[k])
                                        + "_t" + std::to_string(t_list[i]);
                //making directory
                std::filesystem::create_directory("result/" + dir_name);
                //Variables for calculating mean and standard deviation values
                double ave = 0.0, std = 0.0;
                //repeat "n" times
                for(int n=0; n < ntimes; n++) {
                    FP.clear_result();
                    //begin calculating the value of Frame Potential
                    FP.calculate();
                    //FP.output();
                    //save the log
                    FP.save_result(dir_name + "/n=" + std::to_string(n) + ".csv");
                    //get the calculation results and add them up
                    ave += FP.get_result();
                    std += (FP.get_result() * FP.get_result());
                }
                //calculate the average
                ave /= ntimes;
                //calculate the standard deviation
                std = sqrt(std / ntimes - ave * ave);
                //output the result
                std::cout << "  result : " << ave << "±" << std << std::endl;
                //save the result
                std::ofstream result_file("./result/" + dir_name + "/ave_std.txt");
                result_file << ave << "±" << std << std::endl;
                result_file.close();
            }
        }
    }
    */
    return 0;
}

int rc_test(){
    // CliffordGroup rc(Nq);
    // for(int i=1;i<=3;i++){
    //     ComplexMatrix mat=rc.sample();
    //     std::cerr << mat << std::endl;
    // }

    std::string circ_type = "RC";
    int ntimes=1;
    int Nq=6;
    int depth=-1;
    int t=3;
    int Nsample=100;
    FramePotential FP = { circ_type };
    double eps = 0.0001;
    unsigned int pat = 5;

    std::cout << "value" << std::endl;
    FP.set_paras(Nq, depth, t, Nsample, eps, pat);
    for(int n=0; n < ntimes; n++) { 
        FP.calculate();
    }
    return 0;
}

int main_jackknife(int argc, char *argv[]){
    if(argc!=6){
        printf("invalid argument count %d. [./main [circuitType] [Nq] [depth] [t] [Nsample] ]\n", argc);
        return 0;
    }

    /* Specify the circuit */
    std::string circ_type = argv[1];
    int Nq=atoi(argv[2]);//7
    int depth=atoi(argv[3]);//14;
    int t=atoi(argv[4]);//4;
    int Nsample=atoi(argv[5]);//1000;
    // std::cout << Nq << ", " << depth << ", " << t << ", " << Nsample << ", " << circ_type << std::endl;

    if(Nq>10 || depth>30){
        printf("can't allocate memory.");
        return 0;
    }

    double eps = 0.0001;
    unsigned int pat = 5;

    FramePotential FP = { circ_type };

    FP.set_paras(Nq, depth, t, Nsample, eps, pat);
    FP.set_culculateType(CulculateType::B12_Jackknife);
    FP.calculate();

    return 0;
}

int main_B12(int argc, char *argv[]) {
    if(argc!=7){
        printf("invalid argument count %d. [./main [circuitType] [Ntimes] [Nq] [depth] [t] [Nsample] ]", argc);
        return 0;
    }

    /* Specify the circuit */
    std::string circ_type = argv[1];
    // std::string circ_type = "LRC";
    // std::string circ_type = "RDC";
    int ntimes=atoi(argv[2]);//1000;
    int Nq=atoi(argv[3]);//7
    int depth=atoi(argv[4]);//14;
    int t=atoi(argv[5]);//4;
    int Nsample=atoi(argv[6]);//100;

    if(ntimes>100000 || Nq>10 ){
        printf("can't allocate memory.");
        return 0;
    }

    double eps = 0.0001;
    unsigned int pat = 5;
    
    /* call the class */
    FramePotential FP = { circ_type };

    std::cout << "value" << std::endl;
    FP.set_paras(Nq, depth, t, Nsample, eps, pat);
    FP.set_culculateType(CulculateType::B12);

    //repeat "n" times
    for(int n=0; n < ntimes; n++) {
        //FP.clear_result();
        //begin calculating the value of Frame Potential   
        FP.calculate();
    }

    //std::cout << "\tave:" << ave << ", std:" << std << std::endl;                
    return 0;
}

// int measure_time() {
//     std::chrono::_V2::system_clock::time_point t1, t2;
//     std::chrono::nanoseconds dur;
//     int64_t msec;
//     t1 = std::chrono::system_clock::now();
//     t2 = std::chrono::system_clock::now();
//     dur = t2 - t1;
//     msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
//     std::cout << "calculate time : " << msec << " milli sec." << std::endl;
// }

int main(int argc, char *argv[]) {
    rc_test();
    // main_jackknife(argc, argv);
    // main_B12(argc, argv);
}