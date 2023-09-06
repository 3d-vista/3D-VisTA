import os
import json
import collections
import numpy as np
from torch.utils.data import Dataset
from dataset.path_config import SCAN_FAMILY_BASE
from copy import deepcopy
import random
import torch
from torch import nn
from utils.label_utils import LabelConverter
from utils.eval_helper import convert_pc_to_box, construct_bbox_corners, eval_ref_one_sample
from dataset.dataset_mixin import LoadScannetMixin, DataAugmentationMixin
 
class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-100):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        return len(self.vocab)    

class ScanQADataset(Dataset, LoadScannetMixin, DataAugmentationMixin):
    def __init__(self, split='train', max_obj_len=60, num_points=1024, pc_type='gt', sem_type='607', filter_lang=False, use_unanswer=True, drop_sample=0.0):
        # make sure all input params is valid
        assert pc_type in ['gt', 'pred']
        assert sem_type in ['607']
        assert split in ['train', 'val', 'test_w_obj', 'test_wo_obj']
        if split == 'train':
            pc_type = 'gt'
        
        # build answer
        train_data = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'annotations/qa/ScanQA_v1.0_' + 'train' + ".json")))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        self.num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        self.answer_vocab = Answer(answer_cands)
        print("total answers is {}".format(self.num_answers))
        
        # load file
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/qa/ScanQA_v1.0_' + split + ".json")
        self.scan_ids = set() # scan ids in data
        self.data = []   # data
        json_data = json.load(open(anno_file, 'r'))
        for item in json_data:
            if use_unanswer or (len(set(item['answers']) & set(answer_cands)) > 0):
                if split == 'train' and drop_sample > 0:
                    rand_number = random.random()
                    if rand_number > drop_sample:
                        self.scan_ids.add(item['scene_id'])
                        self.data.append(item)
                else:
                    self.scan_ids.add(item['scene_id'])
                    self.data.append(item)
               
        print(split + " unanswerable question {}, answerable question {}".format(len(json_data) - len(self.data), len(self.data)))
         
        # fill parameters
        self.split = split
        self.max_obj_len = max_obj_len - 1
        self.num_points = num_points
        self.pc_type = pc_type
        self.sem_type = sem_type
        self.filter_lang = filter_lang
        
        # load category file
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2-labels.combined.tsv"))
        
        # load scans
        self.scans = self.load_scannet(self.scan_ids, self.pc_type, not ('test' in self.split))
        self.is_test = 'test' in self.split
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # load scanqa
        item = self.data[idx]
        item_id = item['question_id']
        scan_id = item['scene_id']
        if not self.is_test:
            tgt_object_id_list = item['object_ids']
            tgt_object_name_list = item['object_names']
            answer_list = item['answers']
            answer_id_list = [self.answer_vocab.stoi(answer) for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
        else:
            tgt_object_id_list = []
            tgt_object_name_list = []
            answer_list = []
            answer_id_list = []
        question = item['question']
       
        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds']) # N, 6
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        elif self.pc_type == 'pred':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds_pred'])
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels_pred'])
            # get obj labels by matching
            if not self.is_test:
                gt_obj_labels = self.scans[scan_id]['inst_labels'] # N
                obj_center = self.scans[scan_id]['obj_center'] 
                obj_box_size = self.scans[scan_id]['obj_box_size']
                obj_center_pred = self.scans[scan_id]['obj_center_pred'] 
                obj_box_size_pred = self.scans[scan_id]['obj_box_size_pred']
                for i in range(len(obj_center_pred)):
                    for j in range(len(obj_center)):
                        if eval_ref_one_sample(construct_bbox_corners(obj_center[j], obj_box_size[j]), construct_bbox_corners(obj_center_pred[i], obj_box_size_pred[i])) >= 0.25:
                            obj_labels[i] = gt_obj_labels[j]
                            break
            
        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']) and (self.int2cat[obj_label] in question)]
                for id in tgt_object_id_list:
                    if id not in selected_obj_idxs:
                        selected_obj_idxs.append(id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
           
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs] 
        
        # build tgt object id and box 
        if self.pc_type == 'gt':
           tgt_object_id_list = [selected_obj_idxs.index(x) for x in tgt_object_id_list]
           tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
           for i in range(len(tgt_object_label_list)):
               assert(self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i])
        elif self.pc_type == 'pred':
            # build gt box
            gt_center = []
            gt_box_size = []
            for cur_id in tgt_object_id_list:
                gt_pcd = self.scans[scan_id]["pcds"][cur_id]
                center, box_size = convert_pc_to_box(gt_pcd)
                gt_center.append(center)
                gt_box_size.append(box_size)
            
            # start filtering 
            tgt_object_id_list = []
            tgt_object_label_list = []
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                for j in range(len(gt_center)):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center[j], gt_box_size[j])) >= 0.25:
                        tgt_object_id_list.append(i)
                        tgt_object_label_list.append(self.cat2int[tgt_object_name_list[j]])
                        break
        assert(len(obj_pcds) == len(obj_labels))

        # crop objects
        if self.max_obj_len < len(obj_labels):
            selected_obj_idxs = tgt_object_id_list.copy()
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in  tgt_object_id_list:
                    if klabel in tgt_object_label_list:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            tgt_object_id_list = [i for i in range(len(tgt_object_id_list))]
            assert len(obj_pcds) == self.max_obj_len
        
        # rebuild tgt_object_id
        if len(tgt_object_id_list) == 0:
            tgt_object_id_list.append(len(obj_pcds))
            tgt_object_label_list.append(5)
            
        # rotate obj
        rot_matrix = self.build_rotate_mat()
                
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # build box
            obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
            obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))
            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)
            
        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)
        
        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]
        
        # convert format
        # answer
        answer_label = torch.zeros(self.num_answers).long()
        for id in answer_id_list:
            answer_label[id] = 1
        # tgt object id
        tgt_object_id = torch.zeros(len(obj_fts) + 1).long() # add 1 for pad place holder
        for id in tgt_object_id_list:
            tgt_object_id[id] = 1
        # tgt object sematic
        if self.sem_type == '607':
            tgt_object_label = torch.zeros(607).long()
        else:
            raise NotImplementedError("semantic type " + self.sem_type) 
        for id in tgt_object_label_list:
            tgt_object_label[id] = 1
        
        data_dict = {
            "sentence": question,
            "scan_id": scan_id,
            "answers": "[answer_seq]".join(answer_list),
            "answer_label": answer_label, # A
            "tgt_object_id": torch.LongTensor(tgt_object_id), # N
            "tgt_object_label": torch.LongTensor(tgt_object_label), # L
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_labels": obj_labels,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id
        }
        
        return data_dict
   
if __name__ == "__main__":
    dataset = ScanQADataset('train')
    print(dataset[0])
    #for i in range(len(dataset)):
    #    dataset[44]
    #    print(i)
    