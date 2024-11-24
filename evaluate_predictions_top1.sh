for dataset in "entailmentbank_validation" "eqasc_test" "esnli_test"
do
    python evaluate_predictions_top1.py --data_prefix results/baseline_malls/beam_size16/${dataset}
done


# dataset="entailmentbank_validation"
# for round in "1" "2" "3" "4" "5"
# do
#     python evaluate_predictions_top1.py --sentence_data results/brio_entailmentbank_round${round}/${dataset}_sentences.jsonl --chain_data data/${dataset}_chains.jsonl
# done

# dataset="eqasc_test"
# for round in "1" "2" "3" "4" "5"
# do
#     python evaluate_predictions_top1.py --sentence_data results/brio_eqasc_round${round}/${dataset}_sentences.jsonl --chain_data data/${dataset}_chains.jsonl
# done


# dataset="esnli_test"
# for round in "1" "2" # "3" "4" "5"
# do
#     python evaluate_predictions_top1.py --sentence_data results/brio_esnli_round${round}/${dataset}_sentences.jsonl --chain_data data/${dataset}_chains.jsonl
# done