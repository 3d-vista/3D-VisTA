import time
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from model.vision.basic_modules import _get_clones, init_weights, generate_mm_casual_mask
from model.vision.transformers import (TransformerDecoderLayer,
                                       TransformerEncoderLayer,
                                       TransformerSpatialDecoderLayer,
                                       TransformerSpatialEncoderLayer)
from pipeline.registry import registry

@registry.register_vision_model("unified_encoder_v2")
class UnifiedSpatialCrossEncoderV2(nn.Module):
    """
       spatial_dim: spatial feature dim, used to modify attention
       dim_loc: 
    """
    def __init__(self, hidden_size=768, num_attention_heads=12, num_layers=4, dim_loc=6):
        super().__init__()
        
        # unfied encoder
        unified_encoder_layer =  TransformerEncoderLayer(hidden_size, num_attention_heads)
        self.unified_encoder = _get_clones(unified_encoder_layer, num_layers)

        # loc layer
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = _get_clones(loc_layer, 1)
        
        # token embedding
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        self.apply(init_weights)

    # default not using caption
    def forward(
        self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks, 
        tgt_object_id=None, caption=False, output_attentions=False, output_hidden_states=False, 
    ):
        txt_len = txt_embeds.shape[1]
        obj_len = obj_embeds.shape[1]
        
        for i, unified_layer in enumerate(self.unified_encoder):
            # add embeddings for points
            query_pos = self.loc_layers[0](obj_locs)
            pc_token_type_ids = torch.ones((obj_embeds.shape[0:2])).long().cuda()
            pc_type_embeds = self.token_type_embeddings(pc_token_type_ids)
            # to do remove query pos
            obj_embeds = obj_embeds + query_pos + pc_type_embeds
            
            # add embeddings for languages
            lang_token_type_ids = torch.zeros((txt_embeds.shape[0:2])).long().cuda()
            lang_type_embeds = self.token_type_embeddings(lang_token_type_ids)
            txt_embeds = txt_embeds + lang_type_embeds
            
            # add tokens to target object
            if caption:
                caption_embeds = torch.zeros((txt_embeds.shape[0], txt_embeds.shape[2])).float().cuda()
                for j in range(txt_embeds.shape[0]):
                    caption_embeds[j] = query_pos[j, tgt_object_id[j].item()]
                txt_embeds += caption_embeds.unsqueeze(1)
            
            # fuse embeddings
            joint_embeds = torch.cat((txt_embeds, obj_embeds), dim=1)
            joint_masks = torch.cat((txt_masks, obj_masks), dim=1)
            
            # transformer
            if caption:
                mm_casual_mask = generate_mm_casual_mask(txt_embeds.shape[1], obj_embeds.shape[1]).cuda()
                joint_embeds, self_attn_matrices = unified_layer(joint_embeds, tgt_mask=mm_casual_mask.logical_not(), tgt_key_padding_mask=joint_masks.logical_not())
            else:
                joint_embeds, self_attn_matrices = unified_layer(joint_embeds, tgt_key_padding_mask=joint_masks.logical_not())
            
            # split
            txt_embeds, obj_embeds = torch.split(joint_embeds, [txt_len, obj_len], dim=1)
        
        return txt_embeds, obj_embeds

if __name__ == '__main__':
    x = UnifiedSpatialCrossEncoderV2().cuda()
    txt_embeds = torch.zeros((3, 10, 768)).cuda()
    txt_masks = torch.ones((3, 10)).cuda()
    obj_embeds = torch.zeros((3, 10, 768)).cuda()
    obj_locs = torch.ones((3, 10, 6)).cuda()
    obj_masks = torch.ones((3, 10)).cuda()
    x(txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks)
    