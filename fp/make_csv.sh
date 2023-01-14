#!/bin/bash
make

# jackknife
# circuit_type=RDC
# Nsample=1000
# for Nq in 2 3 4 5 6 7
# do
#     for depth in 3 4 5 6 7 8 9 10
#     do
#         for t in 1 2 3 4
#         do
#             echo $Nq $depth $t
#             ./main "$circuit_type" "$Nq" "$depth" "$t" "$Nsample" > /mnt/d/quantum/result/jackknife/"$circuit_type"_Nq"$Nq"_depth"$depth"_t"$t"_sample"$Nsample".csv
#         done
#     done
# done

# circuit_type=RC
# Nsample=1000
# for Nq in 2 3 4 5 6 7
# do
#     for t in 1 2 3 4
#     do
#         echo $Nq $t
#         ./main "$circuit_type" "$Nq" "$depth" "$t" "$Nsample" > /mnt/d/quantum/result/jackknife/"$circuit_type"_Nq"$Nq"_t"$t"_sample"$Nsample".csv
#     done
# done


# B12
circuit_type=LRC
Ntimes=40
Nsample=25
# Ntimes=1000
# Nsample=100
for Nq in 2 3 4 5 6 7
do
    for depth in 3 4 5 6 7 8 9 10
    do
        for t in 1 2 3 4
        do
            echo $Nq $depth $t
            ./main "$circuit_type" "$Ntimes" "$Nq" "$depth" "$t" "$Nsample" > /mnt/d/quantum/result/"$Ntimes"times/"$circuit_type"_Nq"$Nq"_"$Ntimes"times_depth"$depth"_t"$t"_sample"$Nsample".csv
        done
    done
done

# circuit_type=RC
# Ntimes=40
# Nsample=25
# # Ntimes=1000
# # Nsample=100
# for Nq in 2 3 4 5 6 7
# do
#     for t in 1 2 3 4
#     do
#         echo $Nq $t
#         ./main RC "$Ntimes" "$Nq" 0 "$t" "$Nsample" > /mnt/d/quantum/result/"$Ntimes"times/RC_Nq"$Nq"_"$Ntimes"times_t"$t"_sample"$Nsample".csv
#     done
# done