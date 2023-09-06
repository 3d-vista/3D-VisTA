import collections
import json
import multiprocessing as mp
import os
import random
from copy import deepcopy

import jsonlines
import numpy as np
import torch
from dataset.dataset_mixin import DataAugmentationMixin, LoadScannetMixin
from dataset.path_config import SCAN_FAMILY_BASE
from torch.utils.data import Dataset
from utils.eval_helper import (construct_bbox_corners, convert_pc_to_box,
                               eval_ref_one_sample)
from utils.label_utils import LabelConverter
from collections import Counter
from scipy import sparse
from dataset.path_config import SCAN_FAMILY_BASE, MASK_BASE
from plyfile import PlyData

class Scan2CapDataset(Dataset, LoadScannetMixin, DataAugmentationMixin):
    def __init__(self, split='train', max_obj_len=60, num_points=1024, pc_type='gt', sem_type='607', filter_lang=False, iou_threshold=0.5):
        # make sure all input params is valid
        # use ground truth for training
        # test can be both ground truth and non-ground truth
        assert pc_type in ['gt', 'pred']
        assert sem_type in ['607']
        assert split in ['train', 'val', 'test']
        if split == 'train':
            pc_type = 'gt'
            
        # load file
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/scanrefer.jsonl')
        split_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/splits/scannetv2_'+ split + ".txt")
        split_scan_ids = set([x.strip() for x in open(split_file, 'r')])
        self.scan_ids = set() # scan ids in data
        self.data = [] # scanrefer data
        corpus_cache = {}
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    scene_id = item['scan_id']
                    object_id = int(item['target_id'])
                    object_name = item['instance_type']
                    key = "{}|{}|{}".format(scene_id, object_id, object_name)
                    if key not in corpus_cache.keys() or split == 'train':
                        self.scan_ids.add(item['scan_id'])
                        self.data.append(item)
                        corpus_cache[key] = True
                    
        # fill parameters
        self.split = split
        self.max_obj_len = max_obj_len - 1
        self.num_points = num_points
        self.pc_type = pc_type
        self.sem_type = sem_type
        self.filter_lang = filter_lang
        self.iou_threshold = iou_threshold
        
        # load category file
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2-labels.combined.tsv"))
        
        # load scans
        self.scans = self.load_scannet(self.scan_ids, self.pc_type, self.split != 'test')
        
        # build unique multiple look up
        for scan_id in self.scan_ids:
            #inst_labels = self.scans[scan_id]['inst_labels']
            cache = {}
            label_list = []
            for item in self.data:
                if item['scan_id'] == scan_id and item['target_id'] not in cache.keys():
                    cache[item['target_id']] = 1
                    label_list.append(self.label_converter.id_to_scannetid[self.cat2int[item['instance_type']]])
            self.scans[scan_id]['label_count'] = Counter(label_list)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # load scanrefer
        item = self.data[idx]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        corpus_key = "{}|{}|{}".format(scan_id, tgt_object_id, tgt_object_name)
        
        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds']) # N, 6
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        elif self.pc_type == 'pred':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds_pred'])
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels_pred'])
            # get obj labels by matching
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
        # do not filter for predicted labels, because these labels are not accurate
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']) and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
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
           tgt_object_id = selected_obj_idxs.index(tgt_object_id)
           tgt_object_label = obj_labels[tgt_object_id]
           tgt_object_id_iou25_list = [tgt_object_id]
           tgt_object_id_iou50_list = [tgt_object_id]
           assert(self.int2cat[tgt_object_label] == tgt_object_name)
        elif self.pc_type == 'pred':
            gt_pcd = self.scans[scan_id]["pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            max_iou = self.iou_threshold
             # find max iou
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center, gt_box_size)) >= max_iou:
                    tgt_object_id = i
                    max_iou = eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center, gt_box_size))
            # find tgt iou 25
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center, gt_box_size)) >= 0.50:
                    tgt_object_id_iou50_list.append(i)
        assert(len(obj_pcds) == len(obj_labels))
        
        # crop objects 
        if self.max_obj_len < len(obj_labels):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
            selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len
            
        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = self.max_obj_len
            
        # rotate obj
        rot_matrix = self.build_rotate_mat()
        
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
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
        
        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long()
        for id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[id] = 1
        for id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[id] = 1
        
        # build unique multiple
        is_multiple = self.scans[scan_id]['label_count'][self.label_converter.id_to_scannetid[tgt_object_label]] > 1
        
        data_dict = {
            "sentence": sentence,
            "corpus_key": corpus_key,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': is_multiple
        }
    
        return data_dict

class Scan2CapTestDataset(Dataset, LoadScannetMixin, DataAugmentationMixin):
    def __init__(self, split='test', max_obj_len=60, num_points=1024, pc_type='pred', sem_type='607', filter_lang=False):
        # make sure all input params is valid
        # use ground truth for training
        # test can be both ground truth and non-ground truth
        assert pc_type in ['pred']
        assert sem_type in ['607']
        assert split in ['test']
            
        # load file
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/scanrefer.jsonl')
        split_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/splits/scannetv2_'+ split + ".txt")
        split_scan_ids = set([x.strip() for x in open(split_file, 'r')])
        self.scan_ids = set() # scan ids in data
        self.data = [] # scanrefer data
        corpus_cache = {}
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    self.scan_ids.add(item['scan_id'])
                    
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
        self.scans = self.load_scannet(self.scan_ids, self.pc_type, self.split != 'test')
        
        # build data
        self.data = []
        for scan_id in self.scan_ids:
            for object_id in range(10):
                self.data.append({'item_id': scan_id + "-" + str(object_id),
                                  'scan_id': scan_id,
                                  'target_id': object_id, 
                                  'instance_type': "object", 
                                  'utterance': ""})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # load scanrefer
        item = self.data[idx]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        corpus_key = "{}|{}|{}".format(scan_id, tgt_object_id, tgt_object_name)
        
        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds']) # N, 6
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        elif self.pc_type == 'pred':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds_pred'])
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels_pred'])
        
        # filter out background or language
        # do not filter for predicted labels, because these labels are not accurate
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']) and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
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
        tgt_object_id = selected_obj_idxs.index(tgt_object_id)
        tgt_object_label = obj_labels[tgt_object_id]
        tgt_object_id_iou25_list = [tgt_object_id]
        tgt_object_id_iou50_list = [tgt_object_id]
        assert(len(obj_pcds) == len(obj_labels))
        
        # crop objects 
        if self.max_obj_len < len(obj_labels):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
            selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len
            
        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = self.max_obj_len
            
        # rotate obj
        rot_matrix = self.build_rotate_mat()
        
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
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
        
        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long()
        for id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[id] = 1
        for id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[id] = 1
        
        
        data_dict = {
            "sentence": sentence,
            "corpus_key": corpus_key,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': False
        }
    
        return data_dict
 
def prepare_corpus(save_file):
    anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/scanrefer.jsonl')
    split_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/splits/scannetv2_'+ 'val' + ".txt")
    split_scan_ids = set([x.strip() for x in open(split_file, 'r')])
    data = {}
    with jsonlines.open(anno_file, 'r') as f:
        for item in f:
            if item['scan_id'] in split_scan_ids:
                scene_id = item['scan_id']
                object_id = int(item['target_id'])
                object_name = item['instance_type']
                key = "{}|{}|{}".format(scene_id, object_id, object_name)
                if key not in data:
                    data[key] = []
                data[key].append('sos ' + " ".join(item['tokens'][0:30]) + ' eos')
    print(list(data)[0])
    torch.save(data, save_file)
