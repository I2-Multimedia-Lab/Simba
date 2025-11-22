import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS


class CrossFormer(nn.Module):  #cross attention/self attention

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)#, batch_first=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x_t = x.transpose(0, 1)
        y_t = y.transpose(0, 1)

        x_attn = self.attn(query=x_t, key=y_t, value=y_t)[0]
        x_attn = x_attn.transpose(0, 1)
        x = short_cut + self.drop_path(x_attn)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x

class LSTNet(nn.Module):
    def __init__(self, out_dim=512):
        super(LSTNet, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])     
        self.crossattention = CrossFormer(1024, 1024, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )

    def forward(self, point_cloud, gt=None):
        l0_xyz = point_cloud
        l0_points = point_cloud

        # get the key points and its features
        keypoints_partial, keyfeatures_partial, _ = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)   
        keyfeatures_partial = self.transformer_1(keyfeatures_partial, keypoints_partial) # B,128,512

        feat_partial = self.expanding(keyfeatures_partial)
        feat_partial = feat_partial.transpose(2, 1).contiguous()
        gf_feat_partial = feat_partial.max(dim=1, keepdim=True)[0]
        feat_partial = torch.cat([feat_partial, gf_feat_partial.repeat(1, feat_partial.size(2), 1)], dim=-1) # B,640,512
  
        l0_xyz_gt = gt
        l0_points_gt = gt

        # get the key points and its features
        keypoints_gt, keyfeatures_gt, _ = self.sa_module_1(l0_xyz_gt, l0_points_gt)  # (B, 3, 512), (B, 128, 512)   
        keyfeatures_gt = self.transformer_1(keyfeatures_gt, keypoints_gt) # B,128,512

        feat_gt = self.expanding(keyfeatures_gt)
        feat_gt = feat_gt.transpose(2, 1).contiguous()
        gf_feat_gt = feat_gt.max(dim=1, keepdim=True)[0]
        feat_gt = torch.cat([feat_gt, gf_feat_gt.repeat(1, feat_gt.size(2), 1)], dim=-1) # B,640,512

        feat=self.crossattention(feat_partial, feat_gt) # B,640,512
        ret = self.mlp(feat)   
        return ret,keypoints_partial


@MODELS.register_module()
class SymmGT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.lstnet = LSTNet(out_dim=512)
        self.loss_func = ChamferDistanceL1()
    
    def get_loss(self, rets, gt):
        loss_coarse = self.loss_func(rets[0], gt)
        return loss_coarse

    def forward(self, point_cloud, gt=None):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        if gt is None:
            raise ValueError("SymmGT requires gt (ground truth) input, but got None.")
        ret, keypoints_partial = self.lstnet(point_cloud.transpose(2,1).contiguous(), gt.transpose(2,1).contiguous() if gt is not None else None)
        b = point_cloud.shape[0]
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints_partial.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        coarse = torch.cat([symmetry_points, keypoints_partial], dim=-1) # B, 1024, 3
        return [coarse.transpose(2,1).contiguous(),coarse.transpose(2,1).contiguous()]
