for type in "further_specification_or_conjunction" "infer_class_from_properties" "inference_from_rule" "property_inheritance" "sequential_inference" "substitution"
do
    for round in "0" "1" "2" "3" "4" "5"
    do
        if [ "$round" -eq 0 ]; then
            sentence_path=results/baseline_malls/beam_size16
        else
            sentence_path=results/brio_entailmentbank_round${round}
        fi
        python evaluate_predictions.py \
            --sentence_data $sentence_path/entailmentbank_validation_sentences.jsonl \
            --chain_data data/entailmentbank_chains/${type}.jsonl \
            --output_prefix results/entailmentbank_reasoning_type/${type}_round${round}
    done
done