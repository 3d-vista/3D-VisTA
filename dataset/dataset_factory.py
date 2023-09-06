import os
from re import L

from dataset.data_wrapper import *
from dataset.scanqa import *
from dataset.scanrefer import *
from dataset.referit3d import *
from dataset.scan2cap import *
from dataset.sqa import SQADataset
from pipeline.registry import registry

@registry.register_dataset("scanrefer")
def get_scanrefer_dataset(split='train', **args):
    dataset = ScanReferDataset(split=split, **args)    
    return dataset

@registry.register_dataset("scanrefer_task")
def get_scanrefer_task_dataset(split='train', tokenizer=None, txt_seq_length=50, pc_seq_length=80, **args):
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = ScanReferDataset(split=split, max_obj_len=pc_seq_length, **args)
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)

@registry.register_dataset("referit3d")
def get_referit3d_dataset(split='train', **args):
    dataset = Referit3DDataset(split=split, **args)    
    return dataset

@registry.register_dataset("referit3d_task")
def get_referit3d_task_dataset(split='train', tokenizer=None, txt_seq_length=50, pc_seq_length=80, **args):
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = Referit3DDataset(split=split, max_obj_len=pc_seq_length, **args)
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)

@registry.register_dataset("scanqa")
def get_scanqa_dataset(split='train', **args):
    dataset = ScanQADataset(split=split, **args)    
    return dataset

@registry.register_dataset("scanqa_task")
def get_scanqa_task_dataset(split='train', tokenizer=None, txt_seq_length=50, pc_seq_length=80, **args):
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = ScanQADataset(split=split, max_obj_len=pc_seq_length, **args)
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)

@registry.register_dataset("sqa")
def get_scanqa_dataset(split='train', **args):
    dataset = SQADataset(split=split, **args)    
    return dataset

@registry.register_dataset("sqa_task")
def get_scanqa_task_dataset(split='train', tokenizer=None, txt_seq_length=50, pc_seq_length=80, **args):
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = SQADataset(split=split, max_obj_len=pc_seq_length, **args)
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)

@registry.register_dataset("caption_task")
def get_caption_task_dataset(split='train', tokenizer=None, vocab=None, corpus=None, txt_seq_length=60, pc_seq_length=80, txt_mask_ratio=0.15, **args):
    if split == 'test':
        dataset = Scan2CapTestDataset(split=split, max_obj_len=pc_seq_length, **args)
    else:
        dataset = Scan2CapDataset(split=split, max_obj_len=pc_seq_length, **args)
    tokenizer = registry.get_language_model(tokenizer)()
    vocab = registry.get_language_model("vocabulary")(vocab)
    return CaptionDatasetWrapper(dataset, tokenizer, vocab, corpus, txt_seq_length, pc_seq_length, txt_mask_ratio, split)
    
if __name__ == '__main__':
    #dataset = get_scanqa_dataset()
    pass
    
