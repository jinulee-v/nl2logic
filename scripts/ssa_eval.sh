for genmode in "beam_size16" # "beam_size1" # "sample_size16_temp1.0" 
do
    for dataset in "entailmentbank_train" "entailmentbank_validation" "enwn_validation" "eqasc_train" "eqasc_validation" "prontoqa_train" "prontoqa_validation"
    do
        python evaluate_predictions.py --data_prefix results/$1/${genmode}/${dataset} &
    done
done