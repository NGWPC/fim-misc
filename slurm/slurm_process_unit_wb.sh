#!/bin/bash

#####################################################################################################################
## Slurm wrapper of fim_process_unit_wb.sh
## Passed a huc list, this script will parallelize the submission of sbatch jobs to the scheduler
## 
## This script must be run AFTER the pre_processing step has completed!
## Do not modify the SBATCH commands in this file.
##
## How to execute:
##      sbatch slurm_process_unit_wb.sh /data/inputs/huc_lists/dev_small_test_4_huc.lst
#####################################################################################################################

## Read number of lines in file supplied as argument
num_lines=$(wc -l $1 | awk '{print $1}')

## Create the Slurm script ($ used in script need to be escaped: \$)
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=placeholder
#SBATCH --output %x_%a.out # %x is the job-name, %a is the Job array ID (index) number.
#SBATCH --partition=compute_1
##SBATCH --ntasks-per-node 10 # Use for more than single-node jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 14 # Use this for cores in single-node jobs.
#SBATCH --time=05:00:00
#SBATCH --array=0-$(( num_lines - 1 ))

## Allow ability to run docker as non-root user 
sudo chmod 666 /var/run/docker.sock

## mkdir -p /fsx/outputs_temp

## Load huc_list.lst file into a bash array
readarray -t HUCS < $1

## Get each individual HUC
HUC=\${HUCS[\$SLURM_ARRAY_TASK_ID]}
export HUC

## Get the run name
RUN_NAME=\${SLURM_JOB_NAME}

echo "Running fim_process_unit_wb.sh on \${HUC}"
echo "RUN_NAME is \${RUN_NAME}"
echo "Container name is \${RUN_NAME}_\${HUC}"

docker run --rm --name \${RUN_NAME}_\${HUC} -v /efs/repo/inundation-mapping/:/foss_fim -v /efs/inputs/:/data/inputs -v /efs/outputs/:/outputs -v /efs/outputs_temp/:/fim_temp fim:latest ./foss_fim/fim_process_unit_wb.sh \${RUN_NAME} \${HUC}

EOF
