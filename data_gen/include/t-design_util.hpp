#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <vector>
#include <numeric>
#include <iomanip>
#include <math.h>

/** Function to generate the diagonal components of a random diagonal unitary matrix
 * Pass to the function "multiply_elementwise_function" in qulacs
 */
inline std::complex<double> gen_RandomDiagonal_element(ITYPE dummy) {
    //calculate exp( (0,2pi] * i )
    Random random;
    return std::exp(std::complex<double>(0.0, random.uniform() * 2 * M_PI));
}

/* Function to create the combination(nCr) */
inline void recursive_set(int n, int r, std::vector<std::vector<int>>& v, std::vector<int>& pattern) {
    /* Explanation of arguments
        int n := number of qubits
        int r := number of bit correlations (k-point bit correlations)
        std::vector v
            := list passed by reference from the function "create_comb_list"
               Store bit combinations in this list
        std::vector pattern
            := list of reference patterns
               Increment this last element to create the combination
               When returning from a function, update it before returning.
    */
    
    /* Explanation of variables in this function
        std::vector temp
            := The pattern is rewritten and pushed to v, but we want to
               continue to use the original array of patterns.
               So we make a copy of the pattern called "temp", modify temp, and push it to v.
        int poped := Value obtained when the end of pattern is popped
        int poped_next := When a pattern is continuously popped, the value of the position to be popped next.
        int poped_thre := Threshold for determining whether to pop a pattern consecutively
    */

    //the lentgh of "pattern" is r => start emumeration
    if (pattern.size() == r) {
        std::vector<int> temp = pattern;
        int poped;
        int pop_next;
        int pop_thre = n - 1;

        //push intto the combination list "v" by incrimenting the last of "pattern"
        for (int i=pattern[r-1];i<n;++i) {
            v.push_back(temp);
            temp[r-1] = i + 1;
        }

        //update "pattern"
        while(true) {
            poped = pattern.back();
            pattern.pop_back();

            //if the size of "pattern" become empty by poping => permutationg combinations shoud have be finished 
            if (pattern.size() == 0) {
                break;
            }
            //if "poped" is "pop_thre" => see next
            else if (poped == pop_thre) {
                pop_next = pattern.back();
                //if "pop_next" is not "pop_thre" => update "pattern" and break
                if (pop_next != pop_thre-1){
                    poped = pattern.back() + 1;
                    pattern.pop_back();
                    pattern.push_back(poped);
                    break;
                }
                //if "pop_next" is "pop_thre-1" => decrement "pop_thre" and repeat this while loop
                else {
                    pop_thre = pop_thre - 1;
                }
            }
            //if "poped" is not "pop_thre" => add 1 to the last value and break
            else {
                poped = pattern.back() + 1;
                pattern.pop_back();
                pattern.push_back(poped);
                break;
            }
        }
    }
    //if the length of "pattern" is smaller than "r" => add to become appropriate initial "pattern" and recurse
    else {
        int last_index = pattern.size()-1;
        pattern.push_back(pattern[last_index]+1);
        recursive_set(n, r, v, pattern);
    }
}

/** Function to create a combination (nCr)
 * When you specify "n" and "r" and pass a 2-dimensional list, it returns all possible combinations in the list.
 */
inline void create_comb_list(int n, int r, std::vector<std::vector<int>>& comb_list) {
    /* Explanations of areguments
    int n := number of qubits
    int r := number of bit correlation(k bit correlation)
    std::vector comb_list := list passed by reference from main function, etc.
                             store bit combinations in this list
    */
    
    /* Explanations of a variable
    p_list := List of a reference pattern.
              Pass this to "recursive_set" and increment the last to make a combination.
              r=1,n can be returned easily
    */

    //if 1-bit correlation => simply return each index as a 2d array
    if (r == 1) {
        std::vector<int> p_list = {0};
        for(int i=0;i<n;++i) {
            comb_list.push_back(p_list);
            p_list[0] = p_list[0] + 1;
        }
    }
    // if n-bit correlation(e.g.5-qubit and 5-bit corr) => simply return a 1d list from 0 to n
    else if (n == r) {
        std::vector<int> p_list;
        for(int i=0;i<n;++i) {
            p_list.push_back(i);
        }
        comb_list.push_back(p_list);
    }
    //if k-point correlations other than the above
    else {
        std::vector<int> p_list;
        for(int i=0;i<r;++i) {
            p_list.push_back(i);
        }
        while (p_list.size() > 0) {
            recursive_set(n, r, comb_list, p_list);
        }
    }
}

/** Function to generate all combinations from r=1 to r=n 
 *  when n is specified for a combination (nCr)
 */
inline std::vector<std::vector<int>> get_possibliyMax_bitCorr(unsigned int num_qubits) {
    std::vector<std::vector<int>> bitCorr_list;
    
    for(int i=1;i<num_qubits+1;++i) {
        create_comb_list(num_qubits, i, bitCorr_list);
    }
    
    return bitCorr_list;
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