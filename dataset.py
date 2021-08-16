from pathlib import Path
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from functools import partial
from rouge import Rouge
from itertools import combinations


rouge = Rouge()

def read_jsonl(path) -> List[Dict[str, List[str]]]:
    buffer = []
    with open(path, "r") as fin:
        for line in fin:
            buffer.append(json.loads(line))
    return buffer

def get_rouge(hypothesis, reference)-> float:
    scores = rouge.get_scores(hypothesis, reference)[0]
    mean_f = sum([value["f"] for value in scores]) / 3
    return mean_f

class CNNDMDataset(Dataset):
    def __init__(self, dataset_file_path: Path, indices_file_path, max_text_length=512, SUMMARY_LENGTH=3, MAX_CANDIDATES=20):
        # I do gonna load it in memory, bc even the full dataset is not that big
        # otherwise package linecache may be handy
        self.data = read_jsonl(dataset_file_path)
        self.ranked_indices = read_jsonl(indices_file_path)
        assert len(self.data) == len(self.ranked_indices), "Dataset file and ranked sents file must have exact number of lines"

        self.SUMMARY_LENGTH = SUMMARY_LENGTH
        self.MAX_CANDIDATES = MAX_CANDIDATES


        tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self.tokenizer = partial(tokenizer, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        text: List[str] = self.data[idx]['text']
        summary: List[str] = self.data[idx]['summary']
        tokenized_text = self.tokenizer(text)
        tokenized_summary = self.tokenizer(summary)
        sentences_by_importance = self.ranked_indices[idx]

        #   - get permutations of desired quantity
        #   - limit it by first MAX_CANDIDATES 
        # if you want to process other datasets, you may need to adjust these numbers according to specific situation.
        possible_summaries = list(combinations(sentences_by_importance, self.SUMMARY_LENGTH))
        possible_summaries = possible_summaries[:self.MAX_CANDIDATES]

        # get ROUGE score for each candidate summary and sort them in descending order
        candidate_rouge_scores = [] # (candidate_index, candidate_text, rouge_score)
        
        for candidate_summary_index in possible_summaries:
            candidate_summary_index = sorted(candidate_summary_index) # ensure natural sentence order
            # get text from summary indices
            candidate_summary_as_text = ""
            for sentence_index in candidate_summary_index:
                sentence = text[sentence_index]
                candidate_summary_as_text += " " + sentence
            # compare it with overall summary
            score = get_rouge(candidate_summary_as_text, summary)
            candidate_rouge_scores.append((candidate_summary_index, candidate_summary_as_text, score))
            
        # SORT THE CANDIDATES
        candidate_rouge_scores.sort(key=lambda x : x[2], reverse=True)
        
        # write candidate indices and score
        candidate_summaries_index_sorted, candidate_summaries_as_text_sorted, candidate_summaries_rouge_sorted = zip(*candidate_rouge_scores) # indices of candidate_id, but sorted by rouge score
        
        candidate_summaries_tokenized_sorted = self.tokenizer(candidate_summaries_as_text_sorted)

        return {
            "text": text, 
            "summary": summary, 
            "tokenized_text": tokenized_text, 
            "tokenized_summary": tokenized_summary,
            "sentence_indices_ranked": sentences_by_importance, 
            "candidate_summaries_index_sorted": candidate_summaries_index_sorted,
            "candidate_summaries_as_text_sorted": candidate_summaries_as_text_sorted,
            "candidate_summaries_rouge_sorted": candidate_summaries_rouge_sorted,
            "candidate_summaries_tokenized_sorted": candidate_summaries_tokenized_sorted
        }
