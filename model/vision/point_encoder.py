import json
import os

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataset.path_config import SCAN_FAMILY_BASE
from model.vision.basic_modules import (_get_clones, calc_pairwise_locs,
                                        get_mlp_head, init_weights, get_mixup_function)
from model.vision.pointnet2.pointnet2_modules import PointnetSAModule
from model.vision.transformers import (TransformerDecoderLayer,
                                       TransformerEncoderLayer,
                                       TransformerSpatialDecoderLayer,
                                       TransformerSpatialEncoderLayer)
from pipeline.registry import registry

def break_up_pc(pc: Tensor):
    """
    Split the pointcloud into xyz positions and features tensors.
    This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)

    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )
    return xyz, features

class PointNetPP(nn.Module):
    """
    Pointnet++ encoder.
    For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
    """

    def __init__(self, sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lens of given hyper-params are not compatible')

        self.encoder = nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(PointnetSAModule(
                npoint=sa_n_points[i],
                nsample=sa_n_samples[i],
                radius=sa_radii[i],
                mlp=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)
        for i in range(len(self.encoder)):
            xyz, features = self.encoder[i](xyz, features)

        return self.fc(features.view(features.size(0), -1))

@registry.register_vision_model("pointnet_point_encoder")
class PcdObjEncoder(nn.Module):
    def __init__(self, path=None, freeze=False):
        super().__init__()

        self.pcd_net = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
        )
        
        self.obj3d_clf_pre_head = get_mlp_head(768, 768, 607, dropout=0.3)
        
        self.dropout = nn.Dropout(0.1)
        
        if path is not None:
            state_dict = torch.load(path)
            d1= [(key.removeprefix("obj_encoder."), val) for key, val in state_dict.items() if key.startswith("obj_encoder")]
            d2 = [(key, val) for key, val in state_dict.items() if key.startswith("obj3d_clf_pre_head")]
            self.load_state_dict(dict(d1 + d2))
        
        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
    
    def freeze_bn(self, m):
        '''Freeze BatchNorm Layers'''
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks):
        if self.freeze:
            self.freeze_bn(self.pcd_net)
            
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.pcd_net(einops.rearrange(obj_pcds, 'b o p d -> (b o) p d') )
        obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        obj_embeds = self.dropout(obj_embeds)
        # freeze
        if self.freeze:
            obj_embeds = obj_embeds.detach()
        # sem logits
        obj_sem_cls = self.obj3d_clf_pre_head(obj_embeds)
        return obj_embeds, obj_embeds, obj_sem_cls

@registry.register_vision_model("point_tokenize_encoder")
class PointTokenizeEncoder(nn.Module):
    def __init__(self, backbone='pointnet++', hidden_size=768, path=None, freeze_feature=False,
                num_attention_heads=12, spatial_dim=5, num_layers=4, dim_loc=6, pairwise_rel_type='center',
                mixup_strategy=None, mixup_stage1=None, mixup_stage2=None):
        super().__init__()
        assert backbone in ['pointnet++', 'pointnext']
        
        # build backbone
        if backbone == 'pointnet++':
            self.point_feature_extractor = PointNetPP(
                sa_n_points=[32, 16, None],
                sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None],
                sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
            )
        elif backbone == 'pointnext':
            self.point_feature_extractor = PointNext()
                      
        # build cls head
        self.point_cls_head = get_mlp_head(hidden_size, hidden_size, 607, dropout=0.0)
        self.dropout = nn.Dropout(0.1) 
        
        # freeze feature
        self.freeze_feature = freeze_feature
        if freeze_feature:
            for p in self.parameters():
                p.requires_grad = False
        
        # build semantic cls embeds
        self.sem_cls_embed_layer = nn.Sequential(nn.Linear(300, hidden_size),
                                                  nn.LayerNorm(hidden_size),
                                                  nn.Dropout(0.1))
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.cat2vec = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/cat2glove42b.json"), 'r'))  
        # build mask embedes
        self.sem_mask_embeddings = nn.Embedding(1, 768)
        # build spatial encoder layer
        pc_encoder_layer = TransformerSpatialEncoderLayer(hidden_size, num_attention_heads, dim_feedforward=2048, dropout=0.1, activation='gelu', 
                                                       spatial_dim=spatial_dim, spatial_multihead=True, spatial_attn_fusion='cond')
        self.spatial_encoder = _get_clones(pc_encoder_layer, num_layers)
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = _get_clones(loc_layer, 1)
        self.pairwise_rel_type = pairwise_rel_type
        self.spatial_dim = spatial_dim
        # build mixup strategy
        self.mixup_strategy = mixup_strategy
        self.mixup_function = get_mixup_function(mixup_strategy, mixup_stage1, mixup_stage2)
        # load weights
        self.apply(init_weights)
        if path is not None:
            self.load_state_dict(torch.load(path), strict=False)
            print('finish load backbone')
        
    
    def freeze_bn(self, m):
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks, obj_labels=None, cur_step=None, max_steps=None):
        if self.freeze_feature:
            self.freeze_bn(self.point_feature_extractor)
        
        # get obj_embdes
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.point_feature_extractor(einops.rearrange(obj_pcds, 'b o p d -> (b o) p d') )
        obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        obj_embeds = self.dropout(obj_embeds)
        if self.freeze_feature:
            obj_embeds = obj_embeds.detach()
            
        # get semantic cls embeds
        obj_sem_cls = self.point_cls_head(obj_embeds) # B, O, 607
        if self.freeze_feature:
            obj_sem_cls = obj_sem_cls.detach()
        if self.mixup_strategy != None:
            obj_sem_cls_mix = self.mixup_function(obj_sem_cls, obj_labels, cur_step, max_steps)
        else:
            obj_sem_cls_mix = obj_sem_cls.clone()
        obj_sem_cls_mix = torch.argmax(obj_sem_cls_mix, dim=2)
        obj_sem_cls_embeds = torch.Tensor([self.cat2vec[self.int2cat[int(i)]] for i in obj_sem_cls_mix.view(batch_size * num_objs)])
        obj_sem_cls_embeds = obj_sem_cls_embeds.view(batch_size, num_objs, 300).cuda()
        obj_sem_cls_embeds = self.sem_cls_embed_layer(obj_sem_cls_embeds)
        obj_embeds = obj_embeds + obj_sem_cls_embeds
        
        # get semantic mask embeds
        obj_embeds = obj_embeds.masked_fill(obj_sem_masks.unsqueeze(2).logical_not(), 0.0)
        obj_sem_mask_embeds = self.sem_mask_embeddings(torch.zeros((batch_size, num_objs)).long().cuda()) * obj_sem_masks.logical_not().unsqueeze(2)
        obj_embeds = obj_embeds + obj_sem_mask_embeds
        
        # record pre embedes
        # note: in ojur implementation, there are three types of embds, raw embeds from PointNet, pre embeds after tokenization, post embeds after transformers
        obj_embeds_pre = obj_embeds
        
        # spatial reasoning
        pairwise_locs = calc_pairwise_locs(obj_locs[:, :, :3], obj_locs[:, :, 3:], pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True, spatial_dim=self.spatial_dim)
        for i , pc_layer in enumerate(self.spatial_encoder):
            query_pos = self.loc_layers[0](obj_locs)
            obj_embeds = obj_embeds + query_pos
            obj_embeds, self_attn_matrices = pc_layer(obj_embeds, pairwise_locs, tgt_key_padding_mask=obj_masks.logical_not())
        
        return obj_embeds, obj_embeds_pre, obj_sem_cls


if __name__ == '__main__':
    x = PointTokenizeEncoder(backbone='pointnet++', hidden_size=768, path="project/pretrain_weights/pointnet_tokenizer.pth", freeze_feature=True).cuda()
    obj_pcds = torch.ones((10, 10, 1024, 6)).float().cuda()
    obj_locs = torch.ones((10, 10, 6)).cuda()
    obj_masks = torch.ones((10, 10)).cuda()
    obj_sem_masks = torch.ones((10, 10)).cuda()
    x(obj_pcds, obj_locs, obj_masks, obj_sem_masks)
    
    
    
    