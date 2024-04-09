#!/bin/bash

#####################################################################################################################
##
## Slurm wrapper of fim_process_unit_wb.sh
## Passed a huc list, this script will parallelize the submission of sbatch jobs to the scheduler
## 
## The key difference between this script and slurm_process_unit_wb.sh is the functionality to split a larger list of
##      of hucs into multiple compute partitions to avoid resource limitations in cloud computing environments. 
##
## This script must be run AFTER the pre_processing step has completed!
## Do not modify the SBATCH commands in this file.
##
## How to execute:
##      bash slurm_partition_process_unit.sh <num_partitions> <run_name> /data/inputs/huc_lists/dev_small_test_4_huc.lst
#####################################################################################################################

## Set the amount of partitions
partitions=$1

## Set the runName
runName=$2

## Get all HUCS into one array
readarray -t HUCS < $3

# Calculate the size of chunks
chunkSize=$(( ${#HUCS[@]} / partitions ))
remainder=$(( ${#HUCS[@]} % partitions ))

printf "chunkSize (hucs per chunk) -> ${chunkSize} \n"
printf "remainder -> ${remainder} \n" 

## Create the subsets arrays of hucs based on amount of partitions
for ((i=0; i<partitions; i++)); do
    start=$((i * chunkSize))
    end=$((start + chunkSize))
    eval "chunked_array_of_hucs_$i=('${HUCS[@]:${start}:${chunkSize}}')"    
done

## Handle the remainder
if [ $remainder -gt 0 ]; then
    start=$((chunkSize * partitions))
    eval "chunked_array_of_hucs_$partitions=('${HUCS[@]:${start}:${remainder}}')"
fi

## Depending on the remainder, iterate over all chunked arrays
if [ $remainder -gt 0 ]; then
    for ((i=0; i<=partitions; i++)); do
        eval "./process_unit_wb_array.sh -p \$i -n \$runName -u \${chunked_array_of_hucs_$i[*]}"
    done
else 
    for ((i=0; i<partitions; i++)); do
        eval "./process_unit_wb_array.sh -p \$i -n \$runName -u \${chunked_array_of_hucs_$i[*]}"
    done
fi
