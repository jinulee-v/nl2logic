import json
import openai
from openai import AzureOpenAI
from tqdm import tqdm
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


def generate_fol_translations(sentence, num_predictions=8):
    # Few-shot examples
    examples = '''Few-shot Example 1:
Sentence: All planets orbit a star.

FOL Translation:
all x. (Planet(x) -> exists y. (Star(y) & Orbits(x, y)))

Few-shot Example 2:
Sentence: Mars is a planet.

FOL Translation:
Planet(mars)

Few-shot Example 3:
Sentence: If a person is a scientist and has access to a laboratory, they can conduct experiments.

FOL Translation:
all x. ((Scientist(x) & HasAccessToLab(x)) -> CanConductExperiments(x))

Few-shot Example 4:
Sentence: Butterflies are insects.

FOL Translation:
all y. (Butterfly(y) -> Insect(y))

Few-shot Example 5:
Sentence: All insects that have wings can fly.

FOL Translation:
all x. (Insect(x) & HasWings(x) -> CanFly(x))
'''

    messages = [
        {"role": "system", "content": '''You will see a natural language sentence. Translate it into first-order logic (FOL).
You MUST use a common set of predicates to represent the meaning in FOL format.

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
        {"role": "user", "content": f"{examples}\n\nTranslate the following sentence into FOL:\n\n'{sentence}'"}
    ]
    
    return run_llm(messages, num_predictions=num_predictions)


def generate_fol_for_all_sentences(sentences_data, num_predictions=8):
    updated_sentences = []
    for sentence_data in tqdm(sentences_data, desc="Generating FOL translations"):
        sentence_nl = sentence_data['nl']
        predictions = generate_fol_translations(sentence_nl, num_predictions=num_predictions)
        sentence_data['prediction'] = predictions  # Add the FOL translations as a list
        updated_sentences.append(sentence_data)
    return updated_sentences

k = 1

# Load the sentences data
sentences_file = 'data/entailmentbank_validation_sentences.jsonl'
sentences_data = load_jsonl(sentences_file)

# Generate 8 FOL translations for each sentence
updated_sentences_data = generate_fol_for_all_sentences(sentences_data, num_predictions=k)

# Save the updated sentences back to a new JSONL file
output_file = f'baseline/entailmentbank_validation_sentences_with_{k}_predictions.jsonl'
save_jsonl(updated_sentences_data, output_file)

print(f"Updated sentences with multiple FOL translations saved to {output_file}")
