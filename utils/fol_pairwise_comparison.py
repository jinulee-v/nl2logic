import json
import re

def extract_number(s):
    match = re.search(r'_(\d+)$', s)
    if match:
        return int(match.group(1))
    return None  # Return None if no number is found


def get_prompt_from_bank(bank, prompt_number):
    file = open(bank, "r")
    
    counter = 1
    for line in file:
        if counter == prompt_number:
            val = json.loads(line)
            
            return val['nl']
        
        counter += 1

def get_pairwise_comps(path, bank):
        
    file = open(path, "r")

    # read each line in the file as a separate JSON object and store in array

    json_list = []

    for line in file:
        json_list.append(json.loads(line))

    
    pairs = []

    for json_1 in json_list:
        for json_2 in json_list:
            if not json_1['score'] == json_2['score'] and json_1['id'] == json_2['id']:
                if json_1['score'] > json_2['score']:
                    pairs.append((json_1, json_2))
                else:
                    pairs.append((json_2, json_1))
                    

    # generate prompt, choice, reject
    prompts = []
    chosen = []
    rejected = []

    for pair in pairs:
        prompts.append(get_prompt_from_bank(bank, extract_number(pair[0]['id'])))
        chosen.append(pair[0]['prediction'])
        rejected.append(pair[1]['prediction'])
        
    return prompts, chosen, rejected


#print(get_prompt_from_bank("../results/baseline_malls/sample_size16_temp1.0/enwn_validation_sentences.jsonl",2))

#dir = "../results/baseline_malls/sample_size16_temp1.0/"

#val = get_pairwise_comps(dir + "enwn_validation_entailment_preserving_rate_eval.jsonl", dir + "enwn_validation_sentences.jsonl")
