import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from pipeline.registry import registry


@registry.register_optimizer("refer_loss_v1")
def get_refer_loss_v1(txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, og3d_logits, tgt_object_label, tgt_object_id, obj_labels, obj_masks):
    og3d_loss = F.cross_entropy(og3d_logits, tgt_object_id.squeeze(1))
    txt_cls_loss = F.cross_entropy(txt_cls_logits, tgt_object_label.squeeze(1))
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    total_loss = og3d_loss + txt_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss
    return total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss

@registry.register_optimizer("qa_loss_v1")
def get_qa_loss_v1(txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, og3d_logits, answer_scores, tgt_object_label, tgt_object_id, obj_labels, obj_masks, answer_label):
    og3d_logits = og3d_logits.masked_fill_(og3d_logits == -float('inf'), 0)
    og3d_loss = F.binary_cross_entropy_with_logits(og3d_logits, tgt_object_id.float(), reduction='sum', weight=obj_masks) / float(tgt_object_id.shape[0])
    txt_cls_loss = F.binary_cross_entropy_with_logits(txt_cls_logits, tgt_object_label.float(), reduction='sum') / float(tgt_object_label.shape[0])
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    answer_loss = F.binary_cross_entropy_with_logits(answer_scores, answer_label.float(), reduction='sum') / answer_scores.shape[0]
    total_loss = og3d_loss + txt_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss + answer_loss
    return total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss

@registry.register_optimizer("pretrain_loss_v1")
def get_pretrain_loss_v1(txt_lm_cls_logits, masked_lm_labels, scene_txt_match_logits, replace, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, obj_labels, obj_sem_masks, obj_masks):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    masked_lm_labels.masked_fill_(replace.unsqueeze(1), -1)
    lm_cls_loss = loss_fct(txt_lm_cls_logits.permute(0, 2, 1), masked_lm_labels)
    match_loss = loss_fct(scene_txt_match_logits, replace.long())
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss_mask = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks.logical_not()).sum() / (obj_masks * obj_sem_masks.logical_not()).sum()
    obj_cls_pre_loss_unmask = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks).sum() / (obj_masks * obj_sem_masks).sum()
    obj_cls_post_loss_mask = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks.logical_not()).sum() / (obj_masks * obj_sem_masks.logical_not()).sum()
    obj_cls_post_loss_unmask = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks).sum() / (obj_masks * obj_sem_masks).sum()
    total_loss = lm_cls_loss + match_loss + obj_cls_raw_loss + obj_cls_pre_loss_unmask + obj_cls_post_loss
    return total_loss, lm_cls_loss, match_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, obj_cls_pre_loss_mask, obj_cls_pre_loss_unmask, obj_cls_post_loss_mask, obj_cls_post_loss_unmask

@registry.register_optimizer("sqa_loss_v1")
def get_qa_loss_v1(obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, answer_scores, obj_labels, obj_masks, answer_label):
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    answer_loss = F.binary_cross_entropy_with_logits(answer_scores, answer_label.float(), reduction='sum') / answer_scores.shape[0]
    total_loss = obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss + answer_loss
    return total_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss


@registry.register_optimizer("caption_loss_v1")
def get_caption_loss_v1(txt_lm_cls_logits, masked_lm_labels, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, obj_labels, obj_masks):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    lm_cls_loss = loss_fct(txt_lm_cls_logits.permute(0, 2, 1), masked_lm_labels)
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    total_loss = lm_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss
    return total_loss, lm_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss