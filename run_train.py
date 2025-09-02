from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)
from trainer_train import DSITrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)


def make_compute_metrics(tokenizer, valid_ids, lookup_table):
    def clean_docid(docid):

        replace_map = {
            '❶': '1', '❷': '2', '❸': '3', 
            '❹': '4', '❺': '5', '❻': '6', 
            '❼': '7', '❽': '8', '❾': '9', 
            '❿': '10'
        }
        for special_char, replacement in replace_map.items():
            docid = docid.replace(special_char, replacement)
        return docid

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):

            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            

            filtered_rank_list = []
            for docid in rank_list:
                cleaned_docid = clean_docid(docid)
                
                try:
                    sequence = [int(x) for x in cleaned_docid.split()]
                except ValueError:
                    continue
                
                is_valid_sequence = True
                for i in range(len(sequence)):
                    subsequence = sequence[:i+1]
                    subsequence_str = str(tuple(subsequence))
                
                    if subsequence_str not in lookup_table:
                        is_valid_sequence = False
                        break
                    
                    next_ids = lookup_table.get(subsequence_str, [])
                    if next_ids == [None] and i != len(sequence) - 1:
                        is_valid_sequence = False
                        break
                
                if (is_valid_sequence and 
                    cleaned_docid not in filtered_rank_list and 
                    cleaned_docid in valid_ids):
                    filtered_rank_list.append(cleaned_docid)
            
            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        
        return {
            "Hits@1": hit_at_1 / len(eval_preds.predictions), 
            "Hits@10": hit_at_10 / len(eval_preds.predictions)
        }
    
    return compute_metrics


def main():
    # prefix constrained
    def generate_lookup_table(dataset):
        lookup_table = {}
    
        for idx in range(len(dataset)):
            sample = dataset[idx]
            text_id = sample[1]  
            
            text_id_tokens = [int(x) for x in text_id.split()]
            
            for i in range(len(text_id_tokens)):
                prefix = tuple(text_id_tokens[:i+1])
                
                next_token = text_id_tokens[i+1] if i+1 < len(text_id_tokens) else None
                
                prefix_str = str(prefix)
                if prefix_str not in lookup_table:

                    lookup_table[prefix_str] = [next_token] if next_token is not None else [None]
        
        return lookup_table


    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0: 
        wandb.login()
        wandb.init(project="Experiment_mnq320k", name=training_args.run_name)

    if 'mt5' in run_args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')


    if run_args.task == "MGR-CSC":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)
        
        lookup_table = generate_lookup_table(train_dataset)
        
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        # def restrict_decode_vocab(batch_idx, prefix_beam):
        #     # Flatten the list of token ids, regardless of position
        #     # Ensure that we only return token ids that are present in the batch
        #     return INT_TOKEN_IDS
        def restrict_decode_vocab(batch_idx, prefix_beam):
            current_sequence = prefix_beam.tolist()

            sequence_str = str(tuple(current_sequence))
            candidate_tokens = INT_TOKEN_IDS
            
            if sequence_str in lookup_table:
                next_ids = lookup_table[sequence_str]
                if next_ids != [None]:
                    candidate_tokens = [token_id for token_id in next_ids if token_id in INT_TOKEN_IDS]
            
            return candidate_tokens

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids, lookup_table),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            lookup_table=lookup_table
        )
        trainer.train()


    else:
        raise NotImplementedError("--task should be in 'MGR-CSC' ")

if __name__ == "__main__":
    main()

