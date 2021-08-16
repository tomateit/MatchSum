import argparse
import os
import subprocess
import json
import tempfile
import multiprocessing
from time import time
from datetime import timedelta
import queue
import logging
from itertools import combinations
from typing import Union, List, Dict
from functools import partial

from rouge import Rouge
from pathlib import Path
from transformers import AutoTokenizer

rouge = Rouge()
# scores = rouge.get_scores(hypothesis, reference)
# [
#   {
#     "rouge-1": {
#       "f": 0.4786324739396596,
#       "p": 0.6363636363636364,
#       "r": 0.3835616438356164
#     },
#     "rouge-2": {
#       "f": 0.2608695605353498,
#       "p": 0.3488372093023256,
#       "r": 0.20833333333333334
#     },
#     "rouge-l": {
#       "f": 0.44705881864636676,
#       "p": 0.5277777777777778,
#       "r": 0.3877551020408163
#     }
#   }
# ]

def get_rouge(hypothesis, reference)-> float:
    scores = rouge.get_scores(hypothesis, reference)[0]
    mean_f = sum([value["f"] for value in scores]) / 3
    return mean_f

MAX_LEN = 512

TEMP_PATH = Path("./temp") # path to store some temporary files

original_data, sent_ids = [], []

def load_jsonl(data_path) -> List[Dict]:
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data



def get_candidates(tokenizer, cls: str, sep_id: List[int], idx: int):
    """
     The whole function is based around idx parameter
     which is passed during multiprocessed execution and points on a specific position
     of globally preloaded files of text+summary and sentence index ranking
    """

    

    
    # load data
    data = {}
    data["text"] = original_data[idx]['text']
    data['summary'] = original_data[idx]['summary']
    

    # get candidate summaries FROM PRELOADED FILE
    # here is for CNN/DM: 
    #   - truncate each document into the 5 most important sentences (using BertExt), 
    #   - then select any 2 or 3 sentences to form a candidate summary, so there are C(5,2)+C(5,3)=20 candidate summaries.
    # if you want to process other datasets, you may need to adjust these numbers according to specific situation.
    TOP_MOST_IMPORTANT_SENTENCES = 5
    sent_id = sent_ids[idx]['sent_id'][:TOP_MOST_IMPORTANT_SENTENCES]
    data['ext_idx'] = sent_id # store our ranked sentences

    candidate_summary_indices = list(combinations(sent_id, 2)) + list(combinations(sent_id, 3))
    
    # if the whole article consists only of 1 sentence
    if len(sent_id) < 2:
        candidate_summary_indices = [sent_id]


    # get ROUGE score for each candidate summary and sort them in descending order
    score = [] # (candidate_index, rouge_score)
    target_summary_as_text = " ".join(data["summary"])
    for candidate_summary_index in candidate_summary_indices:
        candidate_summary_index = sorted(candidate_summary_index)
        # get text from summary indices
        candidate_summary_as_text = ""
        for sentence_index in candidate_summary_index:
            sentence = data['text'][sentence_index]
            candidate_summary_as_text += " " + sentence
        # compare it with overall summary
        score = get_rouge(candidate_summary_as_text, target_summary_as_text)
        score.append((candidate_summary_index, score))
    score.sort(key=lambda x : x[1], reverse=True)
    
    
    # write candidate indices and score
    data['indices'] = [] # indices of candidate_id, but sorted by rouge score
    data['score'] = [] # rouge scores matching candidate indices
    for candidate_summary_index, rouge_score in score:
        data['indices'].append(list(map(int, candidate_summary_index)))
        data['score'].append(rouge_score)

    # tokenize candidate summary 
    candidate_summaries_as_text = [] # keeping in mind that these are sorted by rouge
    for candidate_summary_indices in data['indices']:
        cur_summary = [cls]
        for sentence_index in candidate_summary_indices:
            cur_summary += data['text'][sentence_index].split()
        cur_summary = cur_summary[:MAX_LEN]
        cur_summary = ' '.join(cur_summary)
        candidate_summaries_as_text.append(cur_summary)
    
    tokenized_summaries = []
    for summary in candidate_summaries_as_text:
        tokenized_summary = tokenizer(summary)
        tokenized_summaries.append(tokenized_summary)
    data['candidate_id'] = tokenized_summaries
    
    # tokenize texts
    tokenized_text = tokenizer(" ".join(data['text'])
    data['text_id'] = tokenized_text

    # tokenize summary
    tokenized_summary = tokenizer(" ".join(data["summary"]))
    data["summary_id"] = tokenized_summary
    
    # write processed data to temporary file
    processed_path = os.path.join(TEMP_PATH, 'processed')
    with open(os.path.join(processed_path, '{}.json'.format(idx)), 'w') as fout:
        json.dump(data, fout, indent=4) 
    

def get_candidates_mp(args):
    
    # choose tokenizer
    if args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cls, sep = '[CLS]', '[SEP]'
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        cls, sep = '<s>', '</s>'
    sep_id = tokenizer.encode(sep, add_special_tokens=False)

    # load original data and indices
    global original_data, sent_ids
    original_data = load_jsonl(args.data_path)
    sent_ids = load_jsonl(args.index_path)

    n_files = len(original_data)
    assert len(sent_ids) == len(original_data)
    print('total {} documents'.format(n_files))
    os.makedirs(TEMP_PATH)
    processed_path = join(TEMP_PATH, 'processed')
    os.makedirs(processed_path)

    # use multi-processing to get candidate summaries
    start = time()
    print('start getting candidates with multi-processing !!!')
    
    with multiprocessing.Pool() as pool:
        list(pool.imap_unordered(partial(get_candidates, tokenizer, cls, sep_id), range(n_files), chunksize=64))
    
    print('finished in {}'.format(timedelta(seconds=time()-start)))
    
    # write processed data
    print('start writing {} files'.format(n_files))
    for i in range(n_files):
        with open(join(processed_path, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        with open(args.write_path, 'a') as f:
            print(json.dumps(data), file=f)
    
    os.system('rm -r {}'.format(TEMP_PATH))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Process truncated documents to obtain candidate summaries'
    )
    parser.add_argument('--tokenizer', type=str, required=True,
        help='BERT/RoBERTa')
    parser.add_argument('--data_path', type=str, required=True,
        help='path to the original dataset, the original dataset should contain text and summary')
    parser.add_argument('--index_path', type=str, required=True,
        help='indices of the remaining sentences of the truncated document')
    parser.add_argument('--write_path', type=str, required=True,
        help='path to store the processed dataset')

    args = parser.parse_args()
    assert args.tokenizer.lower() in ['bert', 'roberta']
    assert os.path.exists(args.data_path)
    assert os.path.exists(args.index_path)

    get_candidates_mp(args)
