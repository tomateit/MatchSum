import torch
from torch import nn
from transformers import AutoModel






class MatchSum(nn.Module):
    """
    
    """
    def __init__(self, candidate_limit, encoder, hidden_size=512):
        super(MatchSum, self).__init__()
        
        self.hidden_size = hidden_size
        self.candidate_limit = candidate_limit

        self.encoder = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")


    def forward(self, tokenized_text, candidate_id, tokenized_summary):
        
        batch_size = tokenized_text.size(0)
        
        pad_id = 0     # for BERT
        if tokenized_text[0][0] == 0:
            pad_id = 1 # for RoBERTa

        # get document embedding
        input_mask = ~(tokenized_text == pad_id)
        out = self.encoder(tokenized_text, attention_mask=input_mask)[0] # last layer
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        input_mask = ~(tokenized_summary == pad_id)
        out = self.encoder(tokenized_summary, attention_mask=input_mask)[0] # last layer
        summary_emb = out[:, 0, :]
        assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        summary_score = nn.functional.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = nn.functional.cosine_similarity(candidate_emb, doc_emb, dim=-1) # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}

