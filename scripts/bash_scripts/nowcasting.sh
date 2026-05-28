#!/bin/bash

# make 6 different nowcasting runs with different parameters experiments = (0, 1, 2) and targets = (bt, bz)
for experiment in 1 2 3
do
    for target in bt bz
    do  
        file_name="nowcasting_experiment_${experiment}_target_${target}.dat"
        cd /Users/syedraza/cmeuncerpy/scripts/Nowcasting/ # go into the working directory.
        python nowcasting_impl.py $experiment $target $file_name
    done
done