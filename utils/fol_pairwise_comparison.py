import json
import re

# give a string that ends with a hyphen and a number, return that number
# used by get_pairwise_comps to extract the line number from the ids of each NL word
def extract_number(s):
    match = re.search(r'_(\d+)$', s)
    if match:
        return int(match.group(1))
    return None  # Return None if no number is found


# returns the NL strong from the bank of NL strings, as specified by which prompt it is
def get_prompt_from_bank(bank, prompt_number):
    file = open(bank, "r")
    
    counter = 1
    for line in file:
        if counter == prompt_number:
            val = json.loads(line)
            
            return val['nl']
        
        counter += 1

# given the path to the NL bank and FOL bank, return the prompts, chosen FOL's, and rejected FOL's 
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
