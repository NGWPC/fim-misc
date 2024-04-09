#!/bin/bash

#####################################################################################################################
## Slurm implementation of fim_pipeline.sh 
##
## This top level script submits slurm_pre_processing.sh, slurm_process_unit_wb.sh/slurm_partition_process_unit.sh, & 
##      slurm_post_processing.sh as sbatch jobs.
##
## Execution example:
##      ./slurm_pipeline.sh -u /data/inputs/huc_lists/dev_small_test_4_huc.lst -n test_slurm_pipeline -p 5 -o
##
## **Note**
##      This workflow has yet to scale beyond 10 HUC8s. Additional PW configurations and CSP availability
##      concerns may need to be addressed. Further evaluation of On Demand Capacity Reservations, or usage of Spot 
##      instances is necessary. 
##
## For Slurm Job dependencies, see:
##      https://slurm.schedmd.com/sbatch.html#OPT_dependency
#####################################################################################################################

:
usage()
{
    echo "
    Processing of HUC's comes in three steps. You can run 'slurm_pipeline.sh' which will run the three main scripts: 
        'slurm_pre_processing.sh', 'slurm_process_unit_wb.sh'/'slurm_partition_process_unit.sh' & 'slurm_post_processing.sh'.
        These scripts are wrappers of : 'fim_pre_processing.sh', 'fim_process_unit_wb.sh' & 'fim_post_processing.sh'.

    Usage : slurm_pipeline.sh -u <huc8> -n <name_of_your_run>

    All arguments to this script are passed to 'fim_pre_processing.sh'.
    REQUIRED:
      -u/--hucList      : HUC8s to run; more than one HUC8 should be passed in quotes (space delimited).
                            A line delimited file, with a .lst extension, is also acceptable.
                            HUC8s must be present in inputs directory.
      -n/--runName      : A name to tag the output directories and log files (only alphanumeric).

    OPTIONS:
      -h/--help         : Print usage statement.
      -jh/--jobLimit    : Max number of concurrent HUC jobs to run. Default 1 job at time.
      -jb/--jobBranchLimit
                        Max number of concurrent Branch jobs to run. Default 1 job at time.
                        - Note: Make sure that the product of jh and jb plus 2 (jh x jb + 2)
                            does not exceed the total number of cores available.
      -p/--partitions   : The amount of partitions available. Used to 'chunk' the huc list, into a subset of arrays to
                            submit them into different partitions.
                            Before the -p argument is supplied, we need to do a little bit of math. First, it is necessary
                            to know how many HUCs are in the file you're submitting (wc -l <huc_list>.lst). 
                            Based off of that number, ideally you should provide the --partition that is evenly divisible 
                            (or as close as possible) by the amount of HUCs. If there is a remainder, there will be another
                            chunked huc array containing the remaining hucs. Be advised that there will be an additional 
                            partition that is needed (+1 of whatever argument provided) to run the remainder. 
                            If there is a remainder, provide a value to the --partition argument which is one less than the 
                            available partitions in the cluster. Take the following exmaples:
                              list_of_10.lst has 10 HUCs, if you provide a -p of 2, there will be five hucs in each array, 
                                    submitted to 2 compute partitions. :)
                              list_of_10.lst has 10 HUCs, if you provide a -p of 3, there will be 4 huc arrays submitted to
                                    4 compute partitions. 
                                   3 arrays with 3 HUCS, and one array comprising the remaining HUC, totalling 4 arrays and 4 partitions.
      -o                : Overwrite outputs if they already exist.
      -s/--skipPost         : If this param is included, the post processing step will be skipped.


    Running 'slurm_pipeline.sh' is a quicker process than running all three scripts independently; however,
        you can run each slurm script them independently if you like. 
    "
    exit
}

# print usage if agrument is '-h' or '--help'
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

while [ "$1" != "" ]; do
    case $1 in
        -u|--hucList)
            shift
            hucList=$1
            ;;
        -n|--runName)
            shift
            runName=$1
            ;;
        -jb|--jobBranchLimit)
            shift
            jobBranchLimit=$1
            ;;
        -p|--partitions)
            shift
            partitions=$1
            ;;
        -h|--help)
            shift
            usage
            ;;
        -o)
            overwrite=1
            ;;
        -s|--skipPost)
            skipPost=1
            ;;
        *) ;;
    esac
    shift
done

# Print usage if arguments empty
if [ "$hucList" = "" ]; then
    echo "ERROR: Missing -u hucList argument"
    usage
    exit 1
fi
if [ "$runName" = "" ]; then
    echo "ERROR: Missing -n runName argument"
    usage
    exit 1
fi

# Default values
if [ "$jobBranchLimit" = "" ]; then jobBranchLimit=1; fi
if [ -z "$partitions" ]; then partitions=0; fi
if [ -z "$overwrite" ]; then overwrite=0; fi
if [ -z "$skipPost" ]; then skipPost=0; fi

## Exit if incorrect arguments (here we need to provide a huclist that is accessible to the docker container)
if [ "${hucList:0:5}" != "/data" ]; then
    printf "\nERROR: Provide a Huclist in format of /data/inputs/huc_lists, a single huc, or, for multiple HUCS "
    printf "use quotes and space delimited pattern as the first parameter\n"
    exit 1
fi

#####################################################################################################################
## Replace runName in slurm_process_unit.sh with $runName argument
sed -i -e "s/placeholder/$runName/g" slurm_process_unit_wb.sh

#####################################################################################################################
## Replace /data/ in huc_list argument to what is provided as the Cluster's Mount Point (eg /fsx, /fimefs, etc) 
## This depends on what was configured in cluster creation 
## Replace </efs> below if using a different Mount Point 

relativeHucList=${hucList/data/efs}

## Read number of lines (hucs) from huclist 
num_hucs=$(wc -l $relativeHucList | awk '{print $1}')

#####################################################################################################################
## Split up the processing into 3 seperate logical processing steps (pre, compute, post)
## Each step correlates to a batch job/job array sent to the scheduler to ensure they are processed in the correct order
## We are making use of the sbatch --dependency option to wait for each step to finish before moving on to the next one 

## The SLURM_PRE_PROCESSING job sets up the folder structure and environment variables
## The PROCESS_UNIT_WB_ARRAY job is an array job, which parallelizes the HUC8 level processing
## The post processing job runs the post processing steps (modifies rating curves, etc)
#####################################################################################################################

printf "\n Initiating fim_pre_processing job with a runName of: $runName \n\t & hucList: $hucList \n"

SLURM_PRE_PROCESSING=$(sbatch --parsable --partition=pre-processing slurm_pre_processing.sh $hucList $runName $jobBranchLimit $overwrite)

#####################################################################################################################
## Parallelization of HUC Processing

## Depending on if the partition argument is provided, issue the appropriate script
if [ $partitions -eq 0 ]; then
    ## Call slurm_process_unit_wb.sh, and assign its slurm job id to PROCESS_UNIT_WB_ARRAY
    PROCESS_UNIT_WB_ARRAY=$(sbatch --partition=process_unit_array --dependency=afterok:$SLURM_PRE_PROCESSING --parsable slurm_process_unit_wb.sh ${relativeHucList})
    printf "\n PROCESS_UNIT_WB_ARRAY Submitted, Job ID is: $PROCESS_UNIT_WB_ARRAY \n"
    SLURM_PROCESS_UNIT_JOB_ARRAY_ID=$(($PROCESS_UNIT_WB_ARRAY + 1))
    printf "\n SLURM_PROCESS_UNIT_JOB_ARRAY_ID is: $SLURM_PROCESS_UNIT_JOB_ARRAY_ID \n"
else
    ## Set the remainder variable. This is needed in order to appropriately set $SLURM_PROCESS_UNIT_JOB_ARRAY_IDS
    remainder=$(( num_hucs % partitions ))
    ## Call slurm_partition_process_unit.sh, and assign its slurm job id to PROCESS_UNIT_WB_ARRAY
    PROCESS_UNIT_WB_ARRAY=$(sbatch --partition=process_unit_array --dependency=afterok:$SLURM_PRE_PROCESSING --parsable slurm_partition_process_unit.sh ${partitions} ${runName} ${relativeHucList})
    printf "\n PROCESS_UNIT_WB_ARRAY Submitted, Job ID is: $PROCESS_UNIT_WB_ARRAY \n"
    ## Depending on if there is a remainder or not, build the string which will be passed as the 'dependency of array jobs' to post_processing step.
    SLURM_PROCESS_UNIT_JOB_ARRAY_IDS=($PROCESS_UNIT_WB_ARRAY:)
    if [ $remainder -ne 0 ]; then
        for ((i=0; i<=partitions; i++)); do
            SLURM_PROCESS_UNIT_JOB_ARRAY_IDS+=$(($PROCESS_UNIT_WB_ARRAY + $i + 1))
            if [ $i -lt $(( $partitions )) ]; then
                SLURM_PROCESS_UNIT_JOB_ARRAY_IDS+=":"
            fi
        done
    else
        for ((i=0; i<partitions; i++)); do
            SLURM_PROCESS_UNIT_JOB_ARRAY_IDS+=$(($PROCESS_UNIT_WB_ARRAY + $i + 1))
            if [ $i -lt $(( $partitions - 1 )) ]; then
                SLURM_PROCESS_UNIT_JOB_ARRAY_IDS+=":"
            fi
        done
    fi
    printf "\n SLURM_PROCESS_UNIT_JOB_ARRAY_IDS (job dependency string that will be passed to post_processing): ${SLURM_PROCESS_UNIT_JOB_ARRAY_IDS} \n"
fi

#####################################################################################################################
# Wait for 30 seconds for the inital jobs (SLURM_PRE_PROCESSING & PROCESS_UNIT_WB_ARRAY) to be submitted

sleep 30 

#####################################################################################################################
## Wait for the Array job submission to complete.
## There are no "futures" in slurm. The job id passed to --dependency must precede the current job id.

## PROCESS_UNIT_WB_ARRAY will be submitted, but it will be in the PD state initially, due to its dependency on
## SLURM_PRE_PROCESSING. Therefore, we are not concerned with SLURM_PRE_PROCESSING, but we do need to wait on PROCESS_UNIT_WB_ARRAY.

while true; do
    job_status=$(squeue -j $PROCESS_UNIT_WB_ARRAY -o %t | tail -n 1)
    if [[ $job_status == "PD" ]] || [[ $job_status == "CF" ]]; then
        echo "Job $PROCESS_UNIT_WB_ARRAY is a pending or configuring state : ($job_status)."
        sleep 60 # Wait for 60 seconds before checking again
    elif [[ $job_status == "R" ]] || [[ $job_status == "CD" ]] || [[ $job_status == "CG" ]] || [[ $job_status == "ST" ]]; then
        echo "Job $PROCESS_UNIT_WB_ARRAY is either in a running, completed, or stopped state: ($job_status)."
        sleep 60 # Wait for 60 seconds for slurm to associate the job array id/ids
        break
    else
        echo "Job $PROCESS_UNIT_WB_ARRAY is in an unaccounted for state: $job_status"
        echo "Please cancel this job by executing: scancel $PROCESS_UNIT_WB_ARRAY"
        echo "See https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES "
        exit 1
    fi

done

#####################################################################################################################
## post-processing 
## - compute c5.24xlarge - 96vCPU -> 48 CPU -2 (fim code requirement) = -j 46

if [ $skipPost -eq 0 ] && [ $partitions -eq 0 ]; then
    ./slurm_post_processing.sh ${runName} ${SLURM_PROCESS_UNIT_JOB_ARRAY_ID}
    printf "\n slurm_post_processing.sh submitted, this depends on all of the array jobs completing. \n"
elif [ $skipPost -eq 0 ] && [ $partitions -ne 0 ];then 
    ./slurm_post_processing.sh ${runName} ${SLURM_PROCESS_UNIT_JOB_ARRAY_IDS}
    printf "\n slurm_post_processing.sh submitted, this depends on all of the array jobs completing. \n"
else
    printf "\n slurm_post_processing.sh skipeed, please remember to run the post processing step after all HUCs have finished processing. \n"
fi

#####################################################################################################################

## Revert runName arg back to placeholder
sed -i -e "s/$runName/placeholder/g" slurm_process_unit_wb.sh 

#####################################################################################################################

printf "\n Jobs submitted, see appropriate SLURM.out files in directory where this script was run for details."
printf "\n\t Run squeue to view the job queue. \n\n"
