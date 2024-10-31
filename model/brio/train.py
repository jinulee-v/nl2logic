import argparse
import json
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys; sys.path.append("model/brio")
from model import BRIO

def load_ce(path, tokenizer, batch_size=16, shuffle=False):
    # Load dataset
    data = []
    with open(path, "r") as f:
        for l in f:
            if not l:
                continue
            datum = json.loads(l)
            input_ids = tokenizer(text=datum["input"], truncation=True, return_tensors="pt")["input_ids"][0]
            labels = tokenizer(text=datum["output"], truncation=True, return_tensors="pt")["input_ids"][0]
            data.append({
                "input_ids": input_ids,
                "labels": labels
            })

    def collate_fn(batch):
        input_ids = pad_sequence([d["input_ids"] for d in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence([d["labels"] for d in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
        return input_ids, labels
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def load_l2r(data_prefix, tokenizer, batch_size=16, shuffle=False):
    # Load dataset
    # sentences_data
    sentences_dict = {}
    with open(data_prefix +  "_sentences.jsonl", "r") as f:
        for l in f:
            if not l:
                continue
            datum = json.loads(l)
            sentences_dict[datum["id"]] = {
                "input": datum["nl"],
                "outputs": [],
                "scores": []
            }

    with open(data_prefix +  "_entailment_preserving_rate_eval.jsonl", "r") as f:
        for l in f:
            if not l:
                continue
            datum = json.loads(l)
            sentences_dict[datum["id"]]["outputs"].append(datum["prediction"])
            sentences_dict[datum["id"]]["scores"].append(datum["score"])

    data = []
    for k, datum in sentences_dict.items():
        assert len(datum["outputs"]) == len(datum["scores"])
        if not any(datum["scores"]):
            # Zero training signal from these examples
            continue
        data.append({
            "input_ids": tokenizer(text=datum["input"], truncation=True, return_tensors="pt")["input_ids"][0],
            "labels": tokenizer(text=datum["outputs"], truncation=True, padding=True, return_tensors="pt")["input_ids"],
            "scores": torch.tensor(datum["scores"], dtype=torch.float)
        })

    def collate_fn(batch):
        max_seq_len = 0
        for d in batch:
            max_seq_len = max(max_seq_len, d["labels"].size(1))
        for d in batch:
            d["labels"] = torch.cat(
                [
                    d["labels"],
                    torch.full(
                        (d["labels"].size(0), max_seq_len - d["labels"].size(1)),
                        fill_value=tokenizer.pad_token_id,
                        dtype=d["labels"].dtype,
                        device=d["labels"].device
                    )
                ],
                dim=1
            )
            assert d["labels"].size(1) == max_seq_len
        input_ids = pad_sequence([d["input_ids"] for d in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence([d["labels"] for d in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
        scores = pad_sequence([d["scores"] for d in batch], batch_first=True, padding_value=-10000)
        return input_ids, labels, scores
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def main(args):
    # Set device
    device = torch.device("cuda")
    # device = torch.device("cpu")

    # Load model
    print("Load model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    brio_model = BRIO(model, tokenizer).to(device)

    # Load dataset
    print("Load dataset...")
    # Supervised dataset: MALLS
    ce_dataloader = load_ce(args.train_malls, tokenizer=tokenizer, batch_size=args.batch_size_per_device)
    # L2R dataset: args.dataset
    l2r_dataloader = load_l2r(args.train_data_prefix, tokenizer=tokenizer, batch_size=args.batch_size_per_device)
    
    # Supervised dataset: MALLS
    valid_ce_dataloader = load_ce(args.valid_malls, tokenizer=tokenizer, batch_size=args.batch_size_per_device)
    # L2R dataset: args.dataset
    valid_l2r_dataloader = load_l2r(args.valid_data_prefix, tokenizer=tokenizer, batch_size=args.batch_size_per_device)

    # Init optimizer
    optimizer = torch.optim.Adam(brio_model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # Train loop
    for epoch in range(args.epoch):
        print(f"Epoch {epoch + 1}/{args.epoch}")
        epoch_len = min(len(ce_dataloader), len(l2r_dataloader))
        optimizer.zero_grad()

        for i, (ce, l2r) in enumerate(tqdm(zip(ce_dataloader, l2r_dataloader), total=epoch_len)):
            # Cross entropy loss
            text, gold = ce
            text, gold = text.to(device), gold.to(device)
            ce_loss = brio_model.ce_loss(text, gold)

            # Learning-to-rank loss
            text, candidates, scores = l2r
            text, candidates, scores = text.to(device), candidates.to(device), scores.to(device)
            l2r_loss = brio_model.l2r_loss(text, candidates, scores)

            loss = (ce_loss + l2r_loss * args.l2r_scale)
            # loss = loss / (1 + args.l2r_scale)
            loss.backward()
            if i == 0:
                continue
            if i % args.grad_accumulation == 0 or i == epoch_len - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if i % args.log_interval == 0 or i == epoch_len - 1:
                print(f"Step {i}, loss: {loss.item() / len(text)}")
            if i % args.save_interval == 0 or i == epoch_len - 1:
                # Validation
                print(f"Validation step: {i} (epoch {epoch})")
                with torch.no_grad():
                    valid_cnt = 0
                    valid_ce_loss, valid_l2r_loss = 0, 0
                    for c, l2r in zip(valid_ce_dataloader, valid_l2r_dataloader):
                        text, gold = c
                        text, gold = text.to(device), gold.to(device)
                        valid_ce_loss += brio_model.ce_loss(text, gold)
                        text, candidates, scores = l2r
                        text, candidates, scores = text.to(device), candidates.to(device), scores.to(device)
                        valid_l2r_loss += brio_model.l2r_loss(text, candidates, scores) * args.l2r_scale
                        valid_cnt += len(text)
                    print(f"Validation loss: {(valid_ce_loss + valid_l2r_loss) / valid_cnt}")
                    brio_model.model.save_pretrained(f"{args.checkpoint_dir}/brio_model_epoch{epoch}_step{i}")
                    brio_model.tokenizer.save_pretrained(f"{args.checkpoint_dir}/brio_model_epoch{epoch}_step{i}")
                    
                    # clear cache
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="baseline/model")
    parser.add_argument("--train_malls", type=str, default="baseline/malls_refined/train.jsonl")
    parser.add_argument("--valid_malls", type=str, default="baseline/malls_refined/validation.jsonl")
    parser.add_argument("--train_data_prefix", type=str, default="results/baseline_malls/beam_size16/entailmentbank_train")
    parser.add_argument("--valid_data_prefix", type=str, default="results/baseline_malls/beam_size16/entailmentbank_validation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/brio")
    
    parser.add_argument("--l2r_scale", type=float, default=10.0)
    
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--batch_size_per_device", type=float, default=4)
    parser.add_argument("--grad_accumulation", type=float, default=4)
    parser.add_argument("--log_interval", type=int, default=40)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--epoch", type=float, default=20)

    args = parser.parse_args()

    main(args)