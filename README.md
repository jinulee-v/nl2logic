# CS546 2024 Fall Team project

Learning generative semantic parsing from entailment labels

## Installation

Runs on Linux, MacOS, and WSL. (Note: due to Vampire, only x86 arch is supported)

```bash
pip install -r requirements.txt
```

## Baseline model (trained from MALLS)

### Train model
```bash
pwd # ../../nl2logic/
cd baseline
python finetune.py # Train on MALLS train set
```

Alternatively, download model from (Deleted link), unzip and place the checkpoint file in `baseline/model/`.
> baseline/model/{model.safetensors, vocab.json, config.json, spiece.model, tokenizer_config.json, ...}

### Evaluate model

```bash
pwd # ../../nl2logic/
cd baseline
python evaluate.py # evaluate on MALLS test set
```

### Generate results for baseline models

```bash
pwd # ../../nl2logic/
scripts/generate_baseline_{beamsearch, greedy, sampling}.sh # Generate results for baseline models
```

## Evaluate using solver

### Prerequisites

If your system is not x86 Linux, build [Vampire](https://vprover.github.io/download.html) from source.

Set the environment variable indicating the path to Vampire executable, `VAMPIRE_PATH="path/to/vampire"`. Default value is `"./vampire"`, and you do not need to modify it if you are running this from x86 Linux.

### Evaluating

You should prepare both `*_chains.jsonl` about the entailment pairs and `*_sentences.jsonl` including predictions for each sentences.

You can run:

```sh
python evaluate_predictions.py --chain_data chains.jsonl --sentence_data sentences.jsonl
```

If your chain and sentences share a common prefix, *e.g.* `results/baseline/entailmentbank_chains.jsonl` and `results/baseline/entailmentbank_sentences.jsonl`, you can run:

```sh
python evaluate_predictions.py --data_prefix results/baseline/entailmentbank.
```