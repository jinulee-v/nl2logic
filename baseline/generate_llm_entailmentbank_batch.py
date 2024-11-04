import json
import openai
from openai import AzureOpenAI
from tqdm import tqdm
from utils.metrics import entailment_preserving_rate_corpus
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


client = openai.OpenAI()

def run_llm(messages):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
    )
    return response.choices[0].message.content


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file]


def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def generate_entailment_fol_translation(premise_sentences, conclusion_sentence):
    premise_text = "\n".join([f"Premise {i+1}: {sentence}" for i, sentence in enumerate(premise_sentences)])
    
    # Few-shot examples
    examples = '''Few-shot Example 1:
Premises: 
Premise 1: All planets orbit a star.
Premise 2: Any celestial body that orbits a star is part of a solar system.
Premise 3: Mars is a planet.
Conclusion: Mars is part of a solar system.

FOL Translation:
all x. (Planet(x) -> exists y. (Star(y) & Orbits(x, y)))
all z. (exists w. (Star(w) & Orbits(z, w)) -> PartOfSolarSystem(z))
Planet(mars)
exists s. (Star(s) & Orbits(mars, s) & PartOfSolarSystem(mars))

Few-shot Example 2:
Premises: 
Premise 1: If a person is a scientist and has access to a laboratory, they can conduct experiments.
Premise 2: All experiments require safety equipment to be performed.
Premise 3: Alice is a scientist with access to a laboratory.
Conclusion: Alice can conduct experiments that require safety equipment.

FOL Translation:
all x. ((Scientist(x) & HasAccessToLab(x)) -> CanConductExperiments(x))
all y. (Experiment(y) -> RequiresSafetyEquipment(y))
Scientist(alice) & HasAccessToLab(alice)
CanConductExperiments(alice) & all e. (Experiment(e) -> RequiresSafetyEquipment(e))

Few-shot Example 3:
Premises: 
Premise 1: All insects that have wings can fly.
Premise 2: Butterflies are insects.
Premise 3: All butterflies have wings.
Conclusion: Butterflies can fly.

FOL Translation:
all x. (Insect(x) & HasWings(x) -> CanFly(x))
all y. (Butterfly(y) -> Insect(y))
all z. (Butterfly(z) -> HasWings(z))
Butterfly(butterfly)
CanFly(butterfly)

Few-shot Example 4:
Premises: 
Premise 1: If a machine is powered on and has sufficient fuel, it will operate.
Premise 2: Engines require fuel to function.
Premise 3: The machine is an engine and is powered on.
Conclusion: The machine will operate.

FOL Translation:
all x. ((PoweredOn(x) & SufficientFuel(x)) -> Operates(x))
all y. (Engine(y) -> RequiresFuel(y))
Engine(machine) & PoweredOn(machine)
SufficientFuel(machine) & Operates(machine)

Few-shot Example 5:
Premises: 
Premise 1: If a person studies hard and has access to study materials, they will pass the exam.
Premise 2: All students preparing for the exam are studying hard.
Premise 3: Emma is a student preparing for the exam.
Conclusion: Emma will pass the exam.

FOL Translation:
all x. ((StudiesHard(x) & HasAccessToMaterials(x)) -> PassesExam(x))
all y. (PreparingForExam(y) -> StudiesHard(y))
Student(emma) & PreparingForExam(emma)
HasAccessToMaterials(emma) & PassesExam(emma)
'''

    messages = [
        {"role": "system", "content": '''You will see some natural language premises followed by a natural language conclusion. Translate them into first-order logic.
You MUST use a common set of predicates to preserve the entailment relation, so that the FOL translations of the premises entail the FOL translation of the conclusion.
You MUST prioritize preserving the entailment relation as opposed to maintaining the semantic meaning. In cases where the meaning of some expressions is similar across premises, you should use the same predicate to represent the meaning.
You MUST strictly output the logical formulas without any labels, one formula per line.
DO NOT include any expository text or empty lines.

Below are instructions for the format of FOL logical formulas:
1. **Variables**: Use lowercase (`x`, `y`, etc.) for generic objects.
2. **Constants**: Use lowercase names (`john`, `sun`) for specific entities.
3. **Predicates**: Represent properties/relations as `Predicate(arg1, arg2)`, e.g., `Rises(sun)`, `Loves(john, mary)`.
4. **Connectives**:
   - **Negation (`-`)**: Not, e.g., `-Rains(x)`
   - **Conjunction (`&`)**: And, e.g., `Walks(john) & Talks(john)`
   - **Disjunction (`|`)**: Or, e.g., `Walks(john) | Talks(john)`
   - **Implication (`->`)**: If...then, e.g., `Rains(x) -> Wet(x)`
   - **Biconditional (`<->`)**: If and only if, e.g., `Rains(x) <-> Wet(x)`
5. **Quantifiers**:
   - **Universal (`all`)**: For all, e.g., `all x. (Human(x) >> Mortal(x))`
   - **Existential (`exists`)**: There exists, e.g., `exists x. (Human(x) & Smart(x))`
6. **Equality (`=`)**: e.g., `john = mary`.'''}, 
        {"role": "user", "content": f"{examples}\n\nAbove are some few shot examples. You need to translate the following natural language premises and conclusion into FOL.\n\nPremises: '{premise_text}'\n\nConclusion: '{conclusion_sentence}'"}
    ]
    
    return run_llm(messages)


def process_entailment_chain(premises, conclusion, sentences_data, entailment):
    # Retrieve NL for each premise
    premise_sentences = []
    for premise_id in premises:
        # Find the premise sentence by ID
        premise_sentence = next((s for s in sentences_data if s['id'] == premise_id), None)
        premise_sentences.append(premise_sentence['nl'] if premise_sentence else '')

    # Retrieve NL for the conclusion
    conclusion_sentence = next((s for s in sentences_data if s['id'] == conclusion), None)
    conclusion_nl = conclusion_sentence['nl'] if conclusion_sentence else ''
    
    # Generate FOL translations for the premises and conclusion in one LLM call
    fol_translation = generate_entailment_fol_translation(premise_sentences, conclusion_nl)
    # print(fol_translation)
    if fol_translation == None:
        return
    fol_lines = fol_translation.split('\n')
    fol_lines = fol_lines[- len(premises) - 1:]
    #print('----------')
    #print(fol_lines)
    
    sentences = []

    # Assign FOL translations to premises and conclusion
    for i, premise_id in enumerate(premises):
        premise_sentence = next((s for s in sentences_data if s['id'] == premise_id), None)
        if premise_sentence:
            premise_sentence['prediction'] = [fol_lines[i] if i < len(fol_lines) else '']  # Assign FOL line to premise
            sentences.append(premise_sentence)

    if conclusion_sentence:
        conclusion_sentence['prediction'] = [fol_lines[len(premises)] if len(fol_lines) > len(premises) else '']  # Assign last FOL line to conclusion
        sentences.append(conclusion_sentence)
    
    score = entailment_preserving_rate_corpus(sentences, [entailment], tqdm=False)
    return score


def generate_entailment_fol(sentences_data, entailment_data):
    score_sum = 0
    total = 0
    for entailment in tqdm(entailment_data):
        premises = entailment['premises']
        conclusion = entailment['conclusion']

        # Process the entailment chain
        score = process_entailment_chain(premises, conclusion, sentences_data, entailment)
        if score == None:
            continue
        score_sum += score[0]
        total += 1

    print(f'Accuracy: {score_sum / total}')
    return sentences_data


# Load the sentences and entailment chains
sentences_file = 'data/entailmentbank_validation_sentences.jsonl'
entailments_file = 'data/entailmentbank_validation_chains.jsonl'

sentences_data = load_jsonl(sentences_file)
entailments_data = load_jsonl(entailments_file)

# Process entailment chains and update sentences with FOL
updated_sentences_data = generate_entailment_fol(sentences_data, entailments_data)

# Save the updated sentences back to a new JSONL file
output_file = 'baseline/entailmentbank_validation_sentences_with_fol_batch.jsonl'
save_jsonl(updated_sentences_data, output_file)

print(f"Updated sentences with FOL translations saved to {output_file}")
