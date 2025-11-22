#!/bin/bash

# Install extensions script with error handling and logging

EXTENSIONS=(
    "chamfer_dist"
    "cubic_feature_sampling"
    "emd"
    "gridding"
    "gridding_loss"
)

LOG_FILE="install_extensions.log"
> $LOG_FILE  # Clear log file

for EXT in "${EXTENSIONS[@]}"; do
    echo "Installing $EXT..." | tee -a $LOG_FILE
    cd extensions/$EXT || { echo "Failed to navigate to extensions/$EXT" | tee -a $LOG_FILE; exit 1; }
    python setup.py install >> $LOG_FILE 2>&1 || { echo "Installation failed for $EXT" | tee -a $LOG_FILE; exit 1; }
    cd - > /dev/null
    echo "$EXT installed successfully." | tee -a $LOG_FILE
done

# Install PointNet++ extension
echo "Installing PointNet++..." | tee -a $LOG_FILE
cd extensions/Pointnet2/pointnet2 || { echo "Failed to navigate to extensions/Pointnet2/pointnet2" | tee -a $LOG_FILE; exit 1; }
python setup.py install >> $LOG_FILE 2>&1 || { echo "Installation failed for PointNet++" | tee -a $LOG_FILE; exit 1; }
cd - > /dev/null
echo "PointNet++ installed successfully." | tee -a $LOG_FILE

echo "All extensions installed successfully." | tee -a $LOG_FILE