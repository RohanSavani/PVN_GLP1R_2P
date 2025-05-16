#!/bin/bash

run_copy() {
    local CMD="$1"
    echo "Starting: $CMD"
    eval "$CMD"
    echo "Finished: $CMD"
}

COMMANDS=(
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/Cue_Data/GLP04_Fasted_L180_P815_550um-02262024-001' './glp4_fasted_cue' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/Cue_Data/GLP04_Fed_L180_P815_550um_022724-009' './glp4_fed_cue' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Cue_Data/GLP06_Fasted_L180_P820_590um-02262024-004' './glp6_fasted_cue' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Cue_Data/GLP06_Fed_L180_P820_590um_022724-011' './glp4_fed_cue' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/glp6_freelickensure30min_L150P800_600um-066_3800licks' './glp6_fastedensure_30min' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/Cue_Data/glp23_fasted_30suc+tone_L155P800_575um-20250220-188' './glp23_fasted_cue' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/Cue_Data/glp23_fed_30suc+tone_L155P800_575um-20250224-214' './glp23_fed_cue' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/fastedsequence/glp23_fasted_30suc_L155P800_575um-20250213-171' './glp23_fasted_d4' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/fastedsequence/glp23_fasted_30suc_L155P800_575um-20250214-185' './glp23_fasted_d5' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/fastedsequence/glp23_fasted_30suc_L155P800_575um-20250211-156' './glp23_fasted_d2' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/fastedsequence/glp23_fasted_30suc_L155P800_575um-20250210-154' './glp23_fasted_d1' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/fastedsequence/glp23_fed_30suc_L155P800_575um-20250212-157' './glp23_fed_d3' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/fastedsequence/glp17_fed_post3dfasted30suc_L150P800_560um-005' './glp17_fed_d4' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/fastedsequence/glp17_fasted30suc_postfed_L150P800_560um-007' './glp17_fasted_d5' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/fastedsequence/glp17_fasted30suc_d3_L150P800_560um-003' './glp17_fasted_d3' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/fastedsequence/glp17_fasted_30sucday2_L150P800_560um-010' './glp17_fasted_d2' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/fastedsequence/glp17_fasted_30suc_d1_L150P800_560um-006' './glp17_fasted_d1' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/glp23_fasted_freelickensure30min_L155P800_575um_31mL_030125-288' './glp23_fastedensure_30min' -vP"
    # "rclone copy 'box:glp17_wd_45suc15quinine(0.5mM)_L150P800_565um-017' './glp17_wd_45suc15quinine(0.5mM)_L150P800_565um-017' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/No Cue Data/GLP06_FedNoCues_L180P815_600um-03062024-1321-073' './GLP06_FedNoCues_L180P815_600um-03062024-1321-073' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/No Cue Data/GLP6_FastedNoCues_L180_P815_600um-_-03092024-1421-084' './GLP6_FastedNoCues_L180_P815_600um-_-03092024-1421-084' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/No Cue Data/GLP6_waterdNoCues_L170_P800_590um-03042024-1137-058' './GLP6_waterdNoCues_L170_P800_590um-03042024-1137-058' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Other Tastants/glp6_wd_L150P800600um_suc0.5mmquinine_032324' './glp6_wd_L150P800600um_suc0.5mmquinine_032324' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Other Tastants/glp6_25sucralose25sucrose_fasted_L170P800_600um_031324-003' './glp6_25sucralose25sucrose_fasted_L170P800_600um_031324-003' -vP"

)


JOB_COUNT=0
MAX_JOBS=2  # Limit number of concurrent jobs

# Loop through the commands and start them in parallel
# for CMD in "${COMMANDS[@]}"; do
#     # Extract directory name from the command using a different method to avoid issues with path extraction
#     DIR=$(echo "$CMD" | sed -n "s/.*'\(.*\)'.*/\1/p" | awk -F' ' '{print $1}')
#     
#     # Create the destination folder if it doesn't exist
#     echo "Creating directory: $DIR"
#     mkdir -p "$DIR"
#     
#     # Run the rclone copy command in the background
#     run_copy "$CMD" &
#     
#     ((JOB_COUNT++))
#     
#     # Limit the number of parallel jobs
#     if (( JOB_COUNT >= MAX_JOBS )); then
#         wait -n  # Wait for any job to finish
#         ((JOB_COUNT--))
#     fi
# done
# 
# # Wait for remaining jobs to finish
# wait
# 
# echo "All downloads completed!"


# Loop through the commands and run them sequentially
for CMD in "${COMMANDS[@]}"; do
    echo "Starting: $CMD"
    
    # Extract directory name from the command
    DIR=$(echo "$CMD" | grep -o "'./[^']*'" | tr -d "'")
    
    # Create the destination folder if it doesn't exist
    mkdir -p "$DIR"
    
    # Run the rclone copy command
    eval "$CMD"
    
    echo "Finished: $CMD"
done

echo "All downloads completed!"


