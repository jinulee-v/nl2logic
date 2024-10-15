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