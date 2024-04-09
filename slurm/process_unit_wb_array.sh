#!/bin/bash

# Set arguments to variables
while [ "$1" != "" ]; do
    case $1 in
        -p|--partition)
            shift
            partition=$1
            ;;
        -n|--runName)
            shift
            runName=$1
            ;;
        -u|--hucList)
            shift
            huc_array=("$@")
            ;;
        *) ;;
    esac
    shift
done

echo
echo "partition: compute_$partition " 
echo "huc_array: ${huc_array[@]}"
echo 

## Create the Slurm script ($ used in script need to be escaped: \$)
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name="${runName}_${partition}"
#SBATCH --output %x_%a.out # %x is the job-name, %a is the Job array ID (index) number.
#SBATCH --partition=compute_$partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 14 # Use this for cores in single-node jobs.
#SBATCH --time=05:00:00
#SBATCH --array=0-$(( ${#huc_array[@]} - 1 ))
##SBATCH --ntasks-per-node 10 # Use for more than single-node jobs

## Allow ability to run docker as non-root user 
sudo chmod 666 /var/run/docker.sock

# Reassign variables
HUCS=(${huc_array[@]})
RUN_NAME=${runName}

HUC=\${HUCS[\$SLURM_ARRAY_TASK_ID]}

echo "Array job number: \${SLURM_ARRAY_TASK_ID}"
echo "Running fim_process_unit_wb.sh on: \${HUC}"
echo "RUN_NAME is \${RUN_NAME}"
echo "Container name is \${HUC}"

docker run --rm --name \${HUC} \
-v /efs/repo/inundation-mapping/:/foss_fim \
-v /efs/inputs/:/data/inputs \
-v /efs/outputs/:/outputs \
-v /efs/outputs_temp/:/fim_temp fim:latest \
./foss_fim/fim_process_unit_wb.sh \${RUN_NAME} \${HUC}

EOF
