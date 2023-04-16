#! /bin/bash

for source_area in 'VISp' 'VISl'
do for target_area in 'VISl' 'VISp'
do run -c 1 -m 300 -t 80:30 -o Out/"$source_area"_"$target_area"_output.out -e Error/"$source_area"_"$target_area"_error.err "python bbp_model_analysis.py --source_area $source_area --target_area $target_area"
done
done

#run -c 1 -m 300 -t 1:30 -o Out/billeh_output.out -e Error/billeh_error.err "python billeh_model_analysis.py"


