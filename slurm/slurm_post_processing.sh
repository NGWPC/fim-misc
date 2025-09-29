#!/bin/bash

#####################################################################################################################
##
##  Slurm wrapper of fim_post_processing.sh
##
##  How to execute:
##      ./slurm_post_processing.sh test_pipeline_steps -1
##
##  Note the -1 as the second argument. This resolves the job_array_id if there are not any - script executed
##      outside of the slurm_pipeline.sh workflow).
##
#####################################################################################################################

run_name=$1
job_array_id=$2

printf "\n Job_array_id from slurm_post_processing.sh is $job_array_id \n"

## Create a secondary Slurm script. This is required in order pass the $job_array_id variable, after it is available, 
## once slurm_process_unit_wb.sh has been submitted
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=slurm_post_processing
#SBATCH --output slurm_post_processing_%j.out
#SBATCH --dependency=afterany:$job_array_id
#SBATCH --partition=post-processing # This is set in PW Cluster Definition
#SBATCH --nodes=1
#SBATCH --cpus-per-task 46 # Use this for threads/cores in single-node jobs - (use same number in slurm_pipeline.sh)
#SBATCH --time=04:00:00

echo "Waited on slurm job array, argument passed to sbatch --dependency: \${SLURM_JOB_DEPENDENCY}"

## Allow ability to run docker as non-root user 
sudo chmod 666 /var/run/docker.sock

RUN_NAME=${run_name}

docker run --rm --name fim_post_processing  \
-v /efs/repo/inundation-mapping/:/foss_fim \
-v /efs/inputs/:/data/inputs \
-v /efs/outputs/:/outputs \
-v /efs/outputs_temp/:/fim_temp fim:latest \
./foss_fim/fim_post_processing.sh -n \${RUN_NAME} -j 46

EOF
