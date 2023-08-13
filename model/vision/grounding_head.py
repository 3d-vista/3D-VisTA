import torch
import torch.nn as nn

import model.vision.pointnet2.pointnet2_utils as pointnet2_utils
from model.vision.basic_modules import get_mlp_head
from pipeline.registry import registry

@registry.register_other_model("ground_head_v1")
class GroundHeadV1(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, sem_cls_size=607, dropout=0.3):
        super().__init__()
        self.og3d_head = get_mlp_head(
            input_size, hidden_size, 
            1, dropout=dropout
        )
        self.txt_clf_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_head = get_mlp_head(
            input_size, hidden_size, 
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_pre_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        
    def forward(self, txt_embeds, obj_embeds, obj_pre_embeds, obj_masks):
        og3d_logits = self.og3d_head(obj_embeds).squeeze(2)
        og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        txt_embeds = txt_embeds.detach()
        obj_embeds = obj_embeds.detach()
        obj_pre_embeds = obj_pre_embeds.detach()
        txt_cls_logits = self.txt_clf_head(txt_embeds[:, 0])
        obj_cls_logits = self.obj3d_clf_head(obj_embeds)
        obj_cls_pre_logits = self.obj3d_clf_pre_head(obj_pre_embeds)
        return txt_cls_logits, obj_cls_logits, obj_cls_pre_logits, og3d_logits
    
if __name__ == '__main__':
    GroundHeadV1()