import torch
import os
from dataset.path_config import SCAN_FAMILY_BASE
import jsonlines
from transformers import BertConfig, BertModel, BertTokenizer
from pipeline.registry import registry

class Vocabulary(object):
    def __init__(self, path=None):
        self.vocab = {}
        self.id_to_vocab = {}
        self.id_to_bert = {}
        
        if path is not None:
            load_dict = torch.load(path)
            self.vocab = load_dict['vocab']
            self.id_to_vocab = load_dict['id_to_vocab']
            self.id_to_bert = load_dict['id_to_bert']
    
    def add_token(self, token, bert_id):
        if token in self.vocab.keys():
            return
        id = len(self.vocab) 
        self.vocab[token] = id
        self.id_to_vocab[id] = token
        self.id_to_bert[id] = bert_id
    
    def token_to_id(self, token):
        return self.vocab[token]
    
    def id_to_token(self, id):
        return self.id_to_vocab[id]
    
    def id_to_bert_id(self, id):
        return self.id_to_bert[id]
        
    def save_vocab(self, path):
        save_dict = {'vocab': self.vocab, "id_to_vocab": self.id_to_vocab, "id_to_bert": self.id_to_bert}
        torch.save(save_dict, path)

def build_scanrefer_bert_vocabulary():
    vocab = Vocabulary()
    vocab.add_token('[EOS]', 102)
    
    anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/scanrefer.jsonl')
    save_path = os.path.join(SCAN_FAMILY_BASE, 'annotations/meta_data/scanrefer_vocab.pth')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  do_lower_case=True)
    
    with jsonlines.open(anno_file, 'r') as f:
        for item in f:
            ids = tokenizer.encode(item['utterance'], add_special_tokens=False)
            for id in ids:
                t = tokenizer.decode([id])
                vocab.add_token(t, id)
    vocab.save_vocab(save_path)
            
@registry.register_language_model("vocabulary")            
def get_vocabulary(path):
    vocab = Vocabulary(path)
    return vocab

