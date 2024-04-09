#!/bin/bash

#####################################################################################################################
##  Use this script for a single HUC submission, a huc list of less than 10 of fim_pipeline.sh, or one very large 
##  compute node (optimized jb and jh values). 
##
##  This job is meant to be configured within this script to meet needs of a particular use case
##     Please modify:
##          SBATCH parameters
##          Arguments to fim_pipeline.sh at the end of the docker run command
##
##  Example:
##      bash slurm_single_fim_pipeline.sh <run_name> <huc8>
#####################################################################################################################

RUN_NAME=$1
HUC_NUMBER=$2

sbatch <<EOF
#!/bin/bash
## The --job-name is transferred to the run_name (output directory)
#SBATCH --job-name=$RUN_NAME
#SBATCH --output %x.out # %x is the job-name
#SBATCH --partition=compute_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10 # Use this for threads/cores in single-node jobs.
#SBATCH --time=06:00:00

# Allow ability to run docker as non-root user 
sudo chmod 666 /var/run/docker.sock

echo "Slurm job name: \${SLURM_JOB_NAME}"

# Make temporary outputs directory if utilizing ephemeral FileSystem
# mkdir -p /fsx/outputs_temp

# Spin up Docker container with correct mounts, and issue fim_pipeline.sh
docker run --rm --name ${RUN_NAME} -v /efs/repo/inundation-mapping/:/foss_fim -v /efs/inputs/:/data/inputs -v /efs/outputs/:/outputs -v /efs/outputs_temp/:/fim_temp fim:latest ./foss_fim/fim_pipeline.sh -u ${HUC_NUMBER} -n ${RUN_NAME} -jh 1 -jb 10 -o

EOF
