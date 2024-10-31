import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def RankingLoss(logprob, scores, margin):
    # Ranking loss from BRIO
    # logprob: [batch_size, beam_size], scores: [batch_size, beam_size]
    assert logprob.size() == scores.size()
    _, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    logprob = torch.gather(logprob, 1, indices)

    ones = torch.ones_like(logprob)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(logprob, logprob, ones)
    # candidate loss
    n = logprob.size(1)
    for i in range(1, n):
        pos_score = logprob[:, :-i]
        neg_score = logprob[:, i:]
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(margin * i)
        loss = loss_func(pos_score, neg_score, ones)
        TotalLoss += loss
    return TotalLoss



class BRIO(nn.Module):
    
    def __init__(self, model, tokenizer): # model = T5
        super(BRIO, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def ce_loss(self, input_ids, labels):
        return self.model(input_ids=input_ids, labels=labels).loss
    
    def l2r_loss(self, input_ids, candidates, scores, margin=0.01):
        logprobs = []
        # obtain sequence probabilities
        for i, decoder_output in zip(input_ids, candidates): # iterate over batch
            # c: tensor([beam_size, seq_len])
            # detect empty sequence
            empty = decoder_output[:, 0] == self.pad_token_id
            # shift right for decoder input
            decoder_input = torch.cat([
                torch.full((decoder_output.size(0), 1), self.pad_token_id, dtype=decoder_output.dtype, device=decoder_output.device),
                decoder_output[:, :-1]
            ], dim=1)
            # calculate sequence probability (average logprob)
            logits = self.model(input_ids=torch.tile(i, (decoder_output.size(0), 1)), decoder_input_ids=decoder_input).logits
            probs = torch.log_softmax(logits, dim=2)
            logprob = torch.gather(probs, 2, decoder_output.unsqueeze(-1)).squeeze(-1) # [beam_size, seq_len]
            mask = decoder_output != self.pad_token_id
            logprob = torch.sum(logprob * mask.float(), dim=1) / (torch.sum(mask.float(), dim=1) + 1e-10) # average logprob, [beam_size]
            # mask out null sequence
            logprob[empty] = -100000
            # append to logprobs
            logprobs.append(logprob.unsqueeze(0)) # [1, beam_size]
        logprobs = torch.cat(logprobs, dim=0) # batch_size, beam_size
        return RankingLoss(logprobs, scores, margin)

    def forward(self, text_id, candidate_id, normalize=True, score_mode="base", length_penalty=1, require_gold=True, adding=0):
        # text_id: [bz, seq_len]
        # candidate_id: [bz, cand_num, seq_len]
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
        cand_mask[:, :, 0] = 1
        output = self.model(
            input_ids=text_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True
            )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty) # [bz, cand_num]
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        return output

    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ):
        return self.model.generate(input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            max_time=max_time,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values,
            synced_gpus=synced_gpus,
            **model_kwargs)