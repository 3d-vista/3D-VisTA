import torch
import math
from tqdm import tqdm
import json
import numpy as np
from utils.cider.cider import Cider
from utils.bleu.bleu import Bleu
from utils.meteor.meteor import Meteor
from utils.rouge.rouge import Rouge
from utils.caption_search import GreedySearch
from utils.box_util import get_3d_box

class NormalDataloaderMixin:
    def __init__(self) -> None:
        pass
    
    def build_dataloader(self, dataset):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size, 
            num_workers=8,
            pin_memory=True,
            shuffle=True, 
            drop_last = True)
        return data_loader

    def build_train_test_dataloader(self):
        if hasattr(self, 'train_dataset'):
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )
        if hasattr(self, 'test_dataset'):
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=False
            )
     
    def prepare_data(self, data_dict):    
        for key in data_dict:
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].cuda()
                
class ModelOptimizationMixin(object):
    def __init__(self):
        pass
    
    @staticmethod
    def warmup_cosine(step, warmup_step, tot_step):
        if step <= warmup_step:
            return step / warmup_step
        return max(0.5 * (1 + math.cos((step - warmup_step) / (tot_step - warmup_step) * math.pi)), 1e-5)
    
    def no_decay_param_group(self, parameters, lr):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        decay_params = []
        no_decay_params = []
        for n, p in parameters:
            if p.requires_grad == False:
                continue
            if not any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optimizer_grouped_parameters = [
            {'params': decay_params,
            'weight_decay': 0.01, 'lr': lr},
            {'params': no_decay_params,
            'weight_decay': 0.0, 'lr': lr}
        ]
        return optimizer_grouped_parameters
        
class ModelEvaluationMixin(object):
    def eval_scanrefer(self, epoch):
        eval_dict = {'target_metric': [], 'og_acc': [], 'og_acc_iou25': [], 'og_acc_iou50': [], 'og_acc_iou25_unique': [], 'og_acc_iou50_unique': [],
                         'og_acc_iou25_multiple': [], 'og_acc_iou50_multiple': [], 'txt_acc': [], 'obj_cls_raw_acc': [], 'obj_cls_pre_acc': [], 'obj_cls_post_acc': []}
         # run
        total_count = 0
        total_unique_count = 0
        total_multiple_count = 0
        if self.eval_task:
            eval_results = []
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            # forward
            data_dict = self.forward_one(data_dict)
            # get metrics
            data_dict = self.get_metrics(data_dict)
            # get count
            count = data_dict['obj_fts'].shape[0]
            unique_count = data_dict['unique_count']
            multiple_count = data_dict['multiple_count']
            total_count += count
            total_unique_count += unique_count
            total_multiple_count += multiple_count
            # save object info
            if self.eval_task:
                og3d_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
                item_ids = data_dict['data_idx']
                for i in range(len(item_ids)):
                    eval_results.append({
                        "scene_id": item_ids[i],
                        "bbox": data_dict['obj_boxes'][i][og3d_pred[i]].cpu().numpy().tolist(), 
                        "correct": bool(data_dict['tgt_object_id_iou25'][i][og3d_pred[i].item()] == 1)
                    })
            #  save eval dict
            for key in eval_dict.keys():
                if 'unique' in key:
                    eval_dict[key].append(float(data_dict[key]) * unique_count)
                elif 'multiple' in key:
                    eval_dict[key].append(float(data_dict[key]) * multiple_count)
                else:
                    eval_dict[key].append(float(data_dict[key]) * count)
        # record
        for k, v in eval_dict.items():
            if 'unique' in k:
                eval_dict[k] = np.sum(v) / total_unique_count
            elif 'multiple' in k:
                eval_dict[k] = np.sum(v) / total_multiple_count
            else:
                eval_dict[k] = np.sum(v) / total_count
        self.record_eval_step(eval_dict, epoch)
        # save results
        if self.eval_task:
            with open('scanrefer_result.json', 'w') as fp:
                json.dump(eval_results, fp)
        return eval_dict['target_metric']

    def eval_referit3d(self, epoch):
        eval_dict = {'target_metric': [], 'og_acc': [], 'og_acc_easy': [], 'og_acc_hard': [], 'og_acc_view_dep': [], 'og_acc_view_indep': [], 'txt_acc': [], 'obj_cls_raw_acc': [], 'obj_cls_pre_acc': [], 'obj_cls_post_acc': []}
         # run
        total_count = 0
        total_easy_count = 0
        total_hard_count = 0
        total_view_dep_count = 0
        total_view_indep_count = 0
        if self.eval_task:
            eval_results = []
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            # forward
            data_dict = self.forward_one(data_dict)
            # get metrics
            data_dict = self.get_metrics(data_dict)
            # get count
            count = data_dict['obj_fts'].shape[0]
            easy_count = data_dict['easy_count']
            hard_count = data_dict['hard_count']
            view_dep_count = data_dict['view_dep_count']
            view_indep_count = data_dict['view_indep_count']
            total_count += count
            total_easy_count += easy_count
            total_hard_count += hard_count
            total_view_dep_count += view_dep_count
            total_view_indep_count += view_indep_count
            # save object info
            if self.eval_task:
                og3d_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
                item_ids = data_dict['data_idx']
                for i in range(len(item_ids)):
                    eval_results.append({
                        "scene_id": item_ids[i],
                        "bbox": data_dict['obj_boxes'][i][og3d_pred[i]].cpu().numpy().tolist(), 
                        "correct": og3d_pred[i].item() == data_dict['tgt_object_id'][i].item()
                    })
            #  save eval dict
            for key in eval_dict.keys():
                if 'easy' in key:
                    eval_dict[key].append(float(data_dict[key]) * easy_count)
                elif 'hard' in key:
                    eval_dict[key].append(float(data_dict[key]) * hard_count)
                elif 'view_dep' in key:
                    eval_dict[key].append(float(data_dict[key]) * view_dep_count)
                elif 'view_indep' in key:
                    eval_dict[key].append(float(data_dict[key]) * view_indep_count)
                else:
                    eval_dict[key].append(float(data_dict[key]) * count)
        # record
        for k, v in eval_dict.items():
            if 'easy' in k:
                eval_dict[k] = np.sum(v) / total_easy_count
            elif 'hard' in k:
                eval_dict[k] = np.sum(v) / total_hard_count
            elif 'view_dep' in k:
                eval_dict[k] = np.sum(v) / total_view_dep_count
            elif 'view_indep' in k:
                eval_dict[k] = np.sum(v) / total_view_indep_count
            else:
                eval_dict[k] = np.sum(v) / total_count
        self.record_eval_step(eval_dict, epoch)
        # save results
        if self.eval_task:
            with open('referit3d_result.json', 'w') as fp:
                json.dump(eval_results, fp)
        return eval_dict['target_metric']
    
    def eval_qa(self, epoch):
        eval_dict = {'target_metric': [], 'og_acc': [], 'txt_acc': [], 'obj_cls_raw_acc': [], 'obj_cls_pre_acc': [], 'obj_cls_post_acc': [], 'ans1_acc': [], 'ans10_acc': []}
        # save results
        if self.eval_task:
            eval_results = []
        # run
        total_count = 0
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            # forward
            data_dict = self.forward_one(data_dict)
            # get metrics
            data_dict = self.get_metrics(data_dict)
            count = data_dict['obj_fts'].shape[0]
            total_count += count
            # save eval results
            if self.eval_task:
                for j in range(count):
                    og3d_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
                    box = data_dict['obj_boxes'][j, og3d_pred[j]].cpu().numpy()
                    box_center = box[0:3]
                    box_size = box[3:6]
                    pred_data = {
                        "scene_id": data_dict["scan_id"][j],
                        "question_id": data_dict["data_idx"][j],
                        "answer_top10": data_dict['answer_top10'][j],
                        "bbox": get_3d_box(box_center, box_size).tolist()
                    
                    }
                    eval_results.append(pred_data)
            # save eval dict
            for key in eval_dict.keys():
                eval_dict[key].append(float(data_dict[key]) * count)
        # record
        for k, v in eval_dict.items():
            eval_dict[k] = np.sum(v) / total_count
        self.record_eval_step(eval_dict, epoch)
        # save eval results
        if self.eval_task:
            with open("scanqa_result.json", "w") as f:
                json.dump(eval_results, f, indent=4)
        return eval_dict['target_metric']
    
    def eval_sqa(self, epoch):
        eval_dict = {'target_metric': [], 'obj_cls_raw_acc': [], 'obj_cls_pre_acc': [], 'obj_cls_post_acc': [], 'ans1_acc': [], 'ans10_acc': [],
                     'type0_acc': [], 'type1_acc': [], 'type2_acc': [], 'type3_acc': [], 'type4_acc': [], 'type5_acc': []}
        # save results
        if self.eval_task:
            eval_results = []
        # run
        total_count = 0
        type_count = {'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10, 'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10}
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            # forward
            data_dict = self.forward_one(data_dict)
            # get metrics
            data_dict = self.get_metrics(data_dict)
            count = data_dict['obj_fts'].shape[0]
            total_count += count
            for key in data_dict:
                if 'type' in key and 'count' in key:
                    type_count[key] += data_dict[key]
            # save eval results
            if self.eval_task:
                for j in range(count):
                    pred_data = {
                        "scene_id": data_dict["scan_id"][j],
                        "question_id": str(data_dict["data_idx"][j].item()),
                        "answer_top10": data_dict['answer_top10'][j]
                    }
                    eval_results.append(pred_data)
            # save eval dict
            for key in eval_dict.keys():
                if 'type' in key:
                    eval_dict[key].append(float(data_dict[key]) * data_dict['type' + key[4] + '_count'])
                else:
                    eval_dict[key].append(float(data_dict[key]) * count)
        # record
        for k, v in eval_dict.items():
            if 'type' in k:
                eval_dict[k] = np.sum(v) / type_count['type' + k[4] + '_count']
            else:
                eval_dict[k] = np.sum(v) / total_count
        self.record_eval_step(eval_dict, epoch)
        # save eval results
        if self.eval_task:
            with open("sqa_result.json", "w") as f:
                json.dump(eval_results, f, indent=4)
        return eval_dict['target_metric']
    
    def eval_caption(self, epoch):
        eval_dict = {'target_metric': 0.0, 'cider': 0.0, 'bleu': 0.0, 'meteor': 0.0, 'rouge': 0.0}
        vocab = self.test_dataset.vocab
        tokenizer = self.test_dataset.tokenizer
        max_seq_length = self.test_dataset.max_seq_length
        gt_sentence_mp = self.test_dataset.corpus
        pred_sentence_mp = {}
        beam_width = 4
        beam_sample_num = 4
        search_type = 'greedy'
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            # build search engine
            search = GreedySearch(data_dict['txt_ids'], tokenizer, vocab, max_seq_length)
            
            # inference with or withou beam search
            while not search.is_end():
                torch.cuda.empty_cache()
                # inference one
                data_dict = self.forward_one(data_dict)
                # convert to id B, 4233
                cur_cls_logit = data_dict['txt_caption_cls_logit'][:, search.cur_id, :].detach()
                search.update(cur_cls_logit)
                data_dict['txt_ids'] = search.get_next_txt_ids() 
                
            # decode ids to tokens
            for j in range(data_dict['gt_ids'].shape[0]):
                # decode predict       
                if search_type == 'beam':
                    if data_dict['tgt_object_id'][j].item() == self.test_dataset.max_obj_len - 1:
                        pred_sentence_mp[data_dict['corpus_key'][j]] = ["sos eos"]
                    else:
                        pred_sentence_mp[data_dict['corpus_key'][j]] = ["sos " + search.output_txt(data_dict['txt_ids'][j * beam_width]) + " eos"]
                else:
                    if data_dict['tgt_object_id'][j].item() == self.test_dataset.max_obj_len - 1:
                        pred_sentence_mp[data_dict['corpus_key'][j]] = ["sos eos"]
                    else:
                        pred_sentence_mp[data_dict['corpus_key'][j]] = ["sos " + search.output_txt(data_dict['txt_ids'][j]) + " eos"]
        # compute cider score
        cider_scorer = Cider()
        bleu_scorer = Bleu()
        meteor_scorer = Meteor()
        rouge_scorer = Rouge()
        if self.test_dataset.split != 'test':
            eval_dict['cider'] = cider_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0]
            eval_dict['bleu'] = bleu_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0][-1]
            eval_dict['meteor'] = meteor_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0]
            eval_dict['rouge'] = rouge_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0]
            eval_dict['target_metric'] =  eval_dict['cider']
            # record
            self.record_eval_step(eval_dict, epoch)
        # save
        if self.eval_task:
            with open('scan2cap_result.json', 'w') as fp:
                json.dump({'gt_sentence_mp': gt_sentence_mp, 'pred_sentence_mp': pred_sentence_mp}, fp)
        return eval_dict

class ModelLossMixin(object):
    def get_refer_loss(self, data_dict):
        total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss = self.refer_loss(data_dict['txt_cls_logits'], data_dict['obj_cls_post_logits'], data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['og3d_logits'], 
                                  data_dict['tgt_object_label'], data_dict['tgt_object_id'], data_dict['obj_labels'], data_dict['obj_masks'])
        data_dict['total_loss'] = total_loss
        data_dict['og3d_loss'] = og3d_loss
        data_dict['txt_cls_loss'] = txt_cls_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        return data_dict
    
    def get_qa_loss(self, data_dict):
        total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss = self.qa_loss(data_dict['txt_cls_logits'], data_dict['obj_cls_post_logits'], data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['og3d_logits'], data_dict['answer_scores'],
                                  data_dict['tgt_object_label'], data_dict['tgt_object_id'], data_dict['obj_labels'], data_dict['obj_masks'], data_dict['answer_label'])
        data_dict['total_loss'] = total_loss
        data_dict['og3d_loss'] = og3d_loss
        data_dict['txt_cls_loss'] = txt_cls_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        data_dict['answer_loss'] = answer_loss
        return data_dict

    def get_sqa_loss(self, data_dict):
        total_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss = self.qa_loss(data_dict['obj_cls_post_logits'], data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['answer_scores'],
                                   data_dict['obj_labels'], data_dict['obj_masks'], data_dict['answer_label'])
        data_dict['total_loss'] = total_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        data_dict['answer_loss'] = answer_loss
        return data_dict
        
    def get_caption_loss(self, data_dict):
        total_loss, caption_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss = self.caption_loss(data_dict['txt_caption_cls_logit'], data_dict['masked_lm_labels'], data_dict['obj_cls_post_logits'], 
                           data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['obj_labels'], data_dict['obj_masks'])
        data_dict['total_loss'] = total_loss
        data_dict['caption_cls_loss'] = caption_cls_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        return data_dict
    
class ModelMetricMixin(object):
    def get_scanrefer_metrics(self, data_dict):
        data_dict['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=1) == data_dict['tgt_object_id'].squeeze(1)).sum().item() / float(len(data_dict['tgt_object_id']))
        # get og acc iou 25 and 50
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
        iou25_correct = 0
        iou50_correct = 0
        iou25_unique_correct = 0
        iou50_unique_correct = 0
        iou25_multiple_correct = 0
        iou50_multiple_correct = 0
        unique_count = 1e-10
        multiple_count = 1e-10
        for i in range(len(og_pred)):
            if data_dict['is_multiple'][i]:
                multiple_count += 1
            else:
                unique_count += 1
            if data_dict['tgt_object_id_iou25'][i, og_pred[i]]:
                iou25_correct += 1
                if data_dict['is_multiple'][i]:
                    iou25_multiple_correct += 1
                else:
                    iou25_unique_correct += 1
            if data_dict['tgt_object_id_iou50'][i, og_pred[i]]:
                iou50_correct += 1
                if data_dict['is_multiple'][i]:
                    iou50_multiple_correct += 1
                else:
                    iou50_unique_correct += 1
        data_dict['og_acc_iou25'] = iou25_correct / float(len(og_pred))
        data_dict['og_acc_iou50'] = iou50_correct / float(len(og_pred))
        data_dict['og_acc_iou25_unique'] = iou25_unique_correct / unique_count
        data_dict['og_acc_iou50_unique'] = iou50_unique_correct / unique_count
        data_dict['og_acc_iou25_multiple'] = iou25_multiple_correct / multiple_count
        data_dict['og_acc_iou50_multiple'] = iou50_multiple_correct / multiple_count
        data_dict['unique_count'] = unique_count
        data_dict['multiple_count'] = multiple_count
        # get other
        data_dict['txt_acc'] = torch.sum(torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).item() / float(len(data_dict['tgt_object_label']))
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['og_acc']
        return data_dict

    def get_referit3d_metrics(self, data_dict):
        data_dict['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=1) == data_dict['tgt_object_id'].squeeze(1)).sum().item() / float(len(data_dict['tgt_object_id']))
        # get og acc iou 25 and 50
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
        easy_correct = 0
        hard_correct = 0
        view_dep_correct = 0
        view_indep_correct = 0
        easy_count = 1e-10
        hard_count = 1e-10
        view_dep_count = 1e-10
        view_indep_count = 1e-10
        for i in range(len(og_pred)):
            if data_dict['is_hard'][i]:
                hard_count += 1
            else:
                easy_count += 1
            if data_dict['is_view_dependent'][i]:
                view_dep_count += 1
            else:
                view_indep_count += 1
            if data_dict['tgt_object_id'][i] == og_pred[i]:
                if data_dict['is_hard'][i]:
                    hard_correct += 1
                else:
                    easy_correct += 1
                if data_dict['is_view_dependent'][i]:
                    view_dep_correct += 1
                else:
                    view_indep_correct += 1
        data_dict['og_acc_easy'] =  easy_correct / easy_count
        data_dict['og_acc_hard'] =  hard_correct / hard_count
        data_dict['og_acc_view_dep'] =  view_dep_correct / view_dep_count
        data_dict['og_acc_view_indep'] =  view_indep_correct / view_indep_count
        data_dict['easy_count'] = easy_count
        data_dict['hard_count'] = hard_count
        data_dict['view_dep_count'] = view_dep_count
        data_dict['view_indep_count'] = view_indep_count
        # get other
        data_dict['txt_acc'] = torch.sum(torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).item() / float(len(data_dict['tgt_object_label']))
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['og_acc']
        return data_dict
        
    def get_qa_metrics(self, data_dict):
        # og, txt
        og3d_argmax = torch.argmax(data_dict['og3d_logits'], dim=1)
        txt_argmax = torch.argmax(data_dict['txt_cls_logits'], dim=1)
        data_dict['og_acc'] = 0.0
        data_dict['txt_acc'] = 0.0
        for i in range(data_dict['tgt_object_id'].shape[0]):
            data_dict['og_acc'] += data_dict['tgt_object_id'][i,og3d_argmax[i]]
            data_dict['txt_acc'] += data_dict['tgt_object_label'][i, txt_argmax[i]]
        data_dict['og_acc'] /= float(data_dict['tgt_object_id'].shape[0])
        data_dict['txt_acc'] /= float(data_dict['tgt_object_label'].shape[0])
        # ans
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'].detach(), 10, -1)[1]
        correct1 = 0
        correct10 = 0
        for i in range(data_dict['answer_label'].shape[0]):
            if data_dict['answer_label'][i, choice_1[i]] == 1:
                correct1 += 1
            for j in range(10):
                if data_dict['answer_label'][i, choice_10[i, j]] == 1:
                    correct10 += 1
                    break
        data_dict['ans1_acc'] = correct1 / float(len(choice_1))
        data_dict['ans10_acc'] = correct10 / float(len(choice_1))
        data_dict['answer_top10'] = [[self.train_dataset.dataset.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)] for i in range(choice_10.shape[0])]
        # cls, cls_pre
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['ans1_acc']
        return data_dict

    def get_sqa_metrics(self, data_dict):
        # ans
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'].detach(), 10, -1)[1]
        correct1 = 0
        correct10 = 0
        correct_type = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4:1e-10, 5:1e-10}
        count_type = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4:1e-10, 5:1e-10}
        for i in range(data_dict['answer_label'].shape[0]):
            count_type[data_dict['sqa_type'][i].item()] += 1
            if data_dict['answer_label'][i, choice_1[i]] == 1:
                correct1 += 1
                correct_type[data_dict['sqa_type'][i].item()] += 1
            for j in range(10):
                if data_dict['answer_label'][i, choice_10[i, j]] == 1:
                    correct10 += 1
                    break
        data_dict['ans1_acc'] = correct1 / float(len(choice_1))
        data_dict['ans10_acc'] = correct10 / float(len(choice_1))
        data_dict['answer_top10'] = [[self.train_dataset.dataset.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)] for i in range(choice_10.shape[0])]
        # cls, cls_pre
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        # question type acc
        for key in count_type.keys():
            data_dict['type' + str(key) + '_acc'] = correct_type[key] / count_type[key]
            data_dict['type' + str(key) + '_count'] = count_type[key]
            
        data_dict['target_metric'] = data_dict['ans1_acc']
        return data_dict

    def get_caption_metrics(self, data_dict):
        # caption acc
        txt_token_mask = (data_dict['masked_lm_labels'] != -1)
        data_dict['caption_cls_acc_mask'] = torch.sum(torch.argmax(data_dict['txt_caption_cls_logit'], dim=2)[txt_token_mask] == data_dict['masked_lm_labels'][txt_token_mask]).item() / float(txt_token_mask.sum().item() + 1e-10)
        # cls, cls_pre
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        return data_dict
   