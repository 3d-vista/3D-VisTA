import torch
import torch.nn.functional as F

class Search(object):
    def output_txt(self, txt_ids):
        txt_len = 1
        while txt_len < self.max_seq_length:
            if txt_ids[txt_len] == 102:
                break
            txt_len += 1
        txt_token_list = txt_ids[1:txt_len]
        decoded_txt = self.tokenizer.decode(txt_token_list)
        # add space
        return ' .'.join(' ,'.join(decoded_txt.split(',')).split("."))
    
    def get_next_txt_ids(self):
        return self.txt_ids
         
class GreedySearch(Search):
    def __init__(self, txt_ids, tokenizer, vocab, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.eos_map = {}
        self.txt_ids = txt_ids.clone()
        self.cur_id = 1
        self.max_seq_length = max_seq_length
        self.batch_size = txt_ids.shape[0]
    
    def update(self, cur_cls_logit):
        cur_cls_id = torch.argmax(cur_cls_logit, dim=1)
        for j in range(cur_cls_logit.shape[0]):
            vocab_id = cur_cls_id[j].item()
            vocab_token = self.vocab.id_to_token(vocab_id)
            bert_id = self.vocab.id_to_bert_id(vocab_id)
            # judge eos
            if vocab_token == '[EOS]':
                self.eos_map[j] = 1
            # fill into txt_ids
            self.txt_ids[j][self.cur_id] = bert_id
        self.cur_id += 1
    
    def is_end(self):
        return len(self.eos_map) == self.batch_size or self.cur_id >= self.max_seq_length
    

    
   
    
    