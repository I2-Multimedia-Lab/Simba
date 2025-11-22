import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS
from .Diffusion.ddim import DDIM
from .Denoiser import UNet_Conv1D_Denoiser
from mamba_ssm import Mamba
from timm.models.layers import DropPath
from .lcm_modules import CrossFormer

def knn(x, k):
    """ Computes the indices of the KNN graph """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """ Gets graph features, used in LCFFN """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Returns (neighbor_feature - center_feature, center_feature)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class LCFFN(nn.Module):
    """ Locally Constrained Feed-Forward Network, a key component of MambaDecoder """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., gcn_k=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Note: The input dimension is in_features * 2 because get_graph_feature concatenates features.
        self.fc1 = nn.Linear(in_features * 2, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.k = gcn_k

    def forward(self, x, center_xyz):
        # x: (B, N, C), center_xyz: (B, N, 3)
        # get_graph_feature requires input of shape (B, C, N)
        # center_xyz should be transposed for knn
        idx = knn(center_xyz.transpose(2, 1), k=self.k)
        # get_graph_feature requires input of shape (B, C, N)
        x_graph = get_graph_feature(x.permute(0, 2, 1), k=self.k, idx=idx).permute(0, 2, 3, 1) # (B, N, K, 2*C)
        
        x_graph = self.fc1(x_graph)
        x_graph = self.act(x_graph)
        x_graph = self.drop(x_graph)
        x_graph = x_graph.max(dim=-2, keepdim=False)[0] # Max-pooling over neighbors
        x_graph = self.fc2(x_graph)
        x_graph = self.drop(x_graph)
        return x_graph

class BlockMamba(nn.Module):
    """ The basic block of the MambaDecoder """
    def __init__(self, dim, mlp_ratio=2., drop_path=0., norm_layer=nn.LayerNorm, gcn_k=5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Mamba(d_model=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LCFFN(in_features=dim, hidden_features=mlp_hidden_dim, gcn_k=gcn_k)

    def forward(self, x, center_xyz):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), center_xyz))
        return x

class BlockMambaCross(nn.Module):
    """
    Achieves information fusion by concatenating sequences and letting a single Mamba module 
    process the long sequence, thus fully preserving the details of the auxiliary sequence y.
    """
    def __init__(self, dim, mlp_ratio=2., drop_path=0., norm_layer=nn.LayerNorm, gcn_k=5):
        super().__init__()
        # Only one standard BlockMamba is needed, which will process the concatenated long sequence
        self.mamba_block = BlockMamba(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            gcn_k=gcn_k
        )

    def forward(self, x, y, center_xyz_x, center_xyz_y):
        """
        Args:
            x (torch.Tensor): The main sequence (Query), shape (B, N, C)
            y (torch.Tensor): The auxiliary sequence (Key/Value), shape (B, M, C)
            center_xyz_x (torch.Tensor): 3D coordinates corresponding to the main sequence x, shape (B, N, 3)
            center_xyz_y (torch.Tensor): 3D coordinates corresponding to the auxiliary sequence y, shape (B, M, 3)
        
        Returns:
            torch.Tensor: The processed main sequence, shape (B, N, C)
        """
        B, N, C = x.shape
        M = y.shape[1]

        xy_cat = torch.cat([x, y], dim=1)
        center_xyz_cat = torch.cat([center_xyz_x, center_xyz_y], dim=1)

        order = center_xyz_cat[:, :, 1].argsort(dim=-1)
        xy_cat_sorted = xy_cat.gather(1, torch.stack([order]*C, -1))
        center_xyz_cat_sorted = center_xyz_cat.gather(1, torch.stack([order]*3, -1))

        processed_cat_sorted = self.mamba_block(xy_cat_sorted, center_xyz_cat_sorted)

        inv_order = order.argsort(dim=-1)
        processed_cat = processed_cat_sorted.gather(1, torch.stack([inv_order]*C, -1))
        
        x_out = processed_cat[:, :N, :]
        
        return x_out
    
class Fusion_hybird(nn.Module):
    def __init__(self, in_channel=256, gcn_k=5):
        super(Fusion_hybird, self).__init__()
        self.crossformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.mamba_self_attention = BlockMamba(in_channel, mlp_ratio=2., gcn_k=gcn_k)

    def forward(self, feat_x, feat_y, center_xyz_x):
        # feat_x: (B, N, C)
        # feat_y: (B, M, C)
        # center_xyz_x: (B, N, 3)
        
        # CrossFormer requires input of shape (B, C, N)
        feat_fused = self.crossformer_1(feat_x, feat_y)
        #print(feat_fused.shape)
        #feat_fused = feat_fused.transpose(1, 2) # -> (B, N, C)

        order = center_xyz_x[:, :, 1].argsort(dim=-1)
        feat_fused_sorted = feat_fused.gather(1, torch.stack([order]*feat_fused.shape[-1], -1))
        center_xyz_x_sorted = center_xyz_x.gather(1, torch.stack([order]*center_xyz_x.shape[-1], -1))

        feat_out_sorted = self.mamba_self_attention(feat_fused_sorted, center_xyz_x_sorted)
        
        inv_order = order.argsort(dim=-1)
        feat_out = feat_out_sorted.gather(1, torch.stack([inv_order]*feat_out_sorted.shape[-1], -1))
        
        return feat_out

class MBA_Refiner_hybird(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(MBA_Refiner_hybird, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)
        
        fusion_dim = 256
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, fusion_dim])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, fusion_dim])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, fusion_dim])

        self.fusion_1 = Fusion_hybird(in_channel=fusion_dim)
        self.fusion_2 = Fusion_hybird(in_channel=fusion_dim)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = BlockMamba(dim=512, mlp_ratio=2., gcn_k=5)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n_coarse = coarse.shape

        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat).transpose(2, 1).contiguous()
        partial_feat = self.expand_dim_2(partial_feat).transpose(2, 1).contiguous()
        symmetry_feat = self.expand_dim_3(symmetry_feat).transpose(2, 1).contiguous()

        coarse_xyz = coarse.transpose(2, 1).contiguous()

        feat_p = self.fusion_1(feat, partial_feat, coarse_xyz)
        feat_s = self.fusion_2(feat, symmetry_feat, coarse_xyz)
        
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)
        
        order = coarse_xyz[:, :, 1].argsort(dim=-1)
        feat_sorted = feat.gather(1, torch.stack([order]*feat.shape[-1], -1))
        coarse_xyz_sorted = coarse_xyz.gather(1, torch.stack([order]*coarse_xyz.shape[-1], -1))
        feat_out_sorted = self.fusion_3(feat_sorted, coarse_xyz_sorted)
        inv_order = order.argsort(dim=-1)
        feat = feat_out_sorted.gather(1, torch.stack([inv_order]*feat_out_sorted.shape[-1], -1))

        offset = self.fc(feat)
        offset = offset.view(b, n_coarse, self.up_factor, 3).reshape(b, n_coarse * self.up_factor, 3)

        pcd_up = coarse_xyz.unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        
        return pcd_up

class MBA_Refiner_PureMamba(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(MBA_Refiner_PureMamba, self).__init__()
        # ... The __init__ part is identical to the one before and requires no modification ...
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.mamba_feat_extractor = BlockMamba(dim=128)
        
        fusion_dim = 256
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, fusion_dim])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, fusion_dim])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, fusion_dim])

        self.fusion_1 = BlockMambaCross(dim=fusion_dim)
        self.fusion_2 = BlockMambaCross(dim=fusion_dim)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = BlockMamba(dim=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )


    def forward(self, coarse, symmetry_feat, partial_feat, symmetry_xyz, partial_xyz):
        """
        Args:
            coarse (torch.Tensor): Input coarse point cloud, containing coordinates and features, shape (B, C, N)
            symmetry_feat (torch.Tensor): Symmetry point features, shape (B, C_feat, M_sym)
            partial_feat (torch.Tensor): Partial point features, shape (B, C_feat, M_par)
            symmetry_xyz (torch.Tensor): Symmetry point coordinates, shape (B, M_sym, 3)
            partial_xyz (torch.Tensor): Partial point coordinates, shape (B, M_par, 3)
        """
        b, _, n_coarse = coarse.shape

        # Extract coordinate information from coarse
        coarse_xyz = coarse[:, :3, :].transpose(1, 2).contiguous() # (B, N, 3)

        # 1. Base feature extraction (logic unchanged)
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        
        feat = feat.transpose(1, 2) # -> (B, N, C)
        order_feat = coarse_xyz[:, :, 1].argsort(dim=-1)
        feat_sorted = feat.gather(1, torch.stack([order_feat]*feat.shape[-1], -1))
        coarse_xyz_sorted = coarse_xyz.gather(1, torch.stack([order_feat]*3, -1))
        feat = self.mamba_feat_extractor(feat_sorted, coarse_xyz_sorted)
        inv_order_feat = order_feat.argsort(dim=-1)
        feat = feat.gather(1, torch.stack([inv_order_feat]*feat.shape[-1], -1))
        feat = feat.transpose(1, 2)

        # 2. Unify dimensions (logic unchanged)
        feat = self.expand_dim_1(feat).transpose(2, 1).contiguous()
        partial_feat = self.expand_dim_2(partial_feat).transpose(2, 1).contiguous()
        symmetry_feat = self.expand_dim_3(symmetry_feat).transpose(2, 1).contiguous()
        
        # 3. Call Mamba Cross Attention (now using coarse_xyz extracted from coarse)
        feat_p = self.fusion_1(feat, partial_feat, coarse_xyz, partial_xyz)
        feat_s = self.fusion_2(feat, symmetry_feat, coarse_xyz, symmetry_xyz)
        
        # 4. Final fusion and self-attention (logic unchanged)
        fused_feat = torch.cat([feat_p, feat_s], dim=-1)
        fused_feat = self.mlp_fusion(fused_feat)
        
        order_final = coarse_xyz[:, :, 1].argsort(dim=-1)
        fused_feat_sorted = fused_feat.gather(1, torch.stack([order_final]*fused_feat.shape[-1], -1))
        coarse_xyz_sorted_final = coarse_xyz.gather(1, torch.stack([order_final]*3, -1))
        fused_feat = self.fusion_3(fused_feat_sorted, coarse_xyz_sorted_final)
        inv_order_final = order_final.argsort(dim=-1)
        fused_feat = fused_feat.gather(1, torch.stack([inv_order_final]*fused_feat.shape[-1], -1))

        # 5. Upsampling (logic unchanged)
        offset = self.fc(fused_feat)
        offset = offset.view(b, n_coarse, self.up_factor, 3).reshape(b, n_coarse * self.up_factor, 3)
        pcd_up = coarse_xyz.unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        
        return pcd_up
class local_encoder(nn.Module):
    def __init__(self,out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self,input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat,torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat,input)
        return feat
    
class PointRefinerMLP(nn.Module):
    """
    A point-wise MLP for refining point cloud coordinates.
    It learns a residual (offset) and adds it to the original input points.
    """
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, point_cloud_xyz):
        """
        Args:
            point_cloud_xyz (torch.Tensor): Input point cloud, shape (B, N, 3).

        Returns:
            torch.Tensor: Refined point cloud, shape (B, N, 3).
        """
        offset = self.mlp(point_cloud_xyz) # Predict the 3D offset for each point
        refined_point_cloud_xyz = point_cloud_xyz + offset  # Add the predicted offset to the original coordinates to achieve refinement   

        return refined_point_cloud_xyz


@MODELS.register_module()
class Simba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]
        self.base_model = MODELS.build(config.base_model)
        self.pretrain_path = config.pretrain if hasattr(config, 'pretrain') else None
        self.pretrained_loaded = False  # 标志是否已加载预训练权重

        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()
        
        self.num_proxy_steps = config.get('num_proxy_steps', 5)  # Default value is 5
        self.use_proxy_refiner = config.get('use_proxy_refiner', False)
        self.local_encoder = local_encoder(out_channel=128)
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256,512])     

        self.denoise_network = UNet_Conv1D_Denoiser(latent_dim=12, time_dim=128, cond_dim=128)
        self.denoise_network = torch.compile(self.denoise_network)

        diffusion_cfg = config.diffusion_cfg.copy()  # Create a copy first
        self.training_mode = diffusion_cfg.pop('training_mode')
        ddim_kwargs = {
            'denoise_model': self.denoise_network,
            'condition_model': nn.Identity(),
            **diffusion_cfg  # Unpack the rest of the parameters from the config
        }
        self.Sym_Diffuser = DDIM(**ddim_kwargs)

        # Compile the MBA_Refiner modules
        self.MBA_Refiner_1 = MBA_Refiner_hybird(gf_dim=512, up_factor=self.up_factors[0]) 
        self.MBA_Refiner_2 = MBA_Refiner_hybird(gf_dim=512, up_factor=self.up_factors[1]) 
        self.MBA_Refiner_3 = MBA_Refiner_PureMamba(gf_dim=512, up_factor=self.up_factors[2])

        self.loss_func = self.build_loss_func()
        # Instantiate the refiner module
        self.point_refiner = PointRefinerMLP(in_dim=3, hidden_dim=128, out_dim=3)

    def _build_coarse_geometry(self, pred_x0, keypoints_partial_xyz):
        """
        Builds a coarse geometry from the predicted x_0.
        
        Args:
            pred_x0 (torch.Tensor): Predicted x_0 (B, N, 12) - contains rotation matrices (9) and translation vectors (3).
            keypoints_partial_xyz (torch.Tensor): Partial point cloud coordinates (B, N, 3).
            
        Returns:
            torch.Tensor: Coarse point cloud (B, 2N, 3).
        """
        b, n_partial, _ = keypoints_partial_xyz.shape
        
        # 🔧 Add dimension check
        if pred_x0.shape != (b, n_partial, 12):
            raise ValueError(f"pred_x0 shape mismatch: expected ({b}, {n_partial}, 12), got {pred_x0.shape}")
        R = pred_x0[:, :, :9].view(b, n_partial, 3, 3)  # (B, N, 3, 3)
        T = pred_x0[:, :, 9:]  # (B, N, 3)
        # keypoints_partial_xyz: (B, N, 3) -> (B, N, 3, 1) for matrix multiplication
        keypoints_expanded = keypoints_partial_xyz.unsqueeze(-1)  # (B, N, 3, 1)
        
        # Calculate symmetry points: R @ keypoints + T
        symmetry_points_xyz = torch.matmul(R, keypoints_expanded).squeeze(-1) + T  # (B, N, 3)
        coarse = torch.cat([keypoints_partial_xyz, symmetry_points_xyz], dim=1)  # (B, 2N, 3)

        return coarse
    
    def build_loss_func(self):
        self.mse_loss = nn.MSELoss()
        self.cd_loss = ChamferDistanceL1()
        
    def get_loss(self, ret, gt, denoised_r, r, epoch=0):
        progress = 1 # 20-epoch warm-up period
        coarse, fine1 , fine2, fine3 = ret
        mse_loss = self.mse_loss(denoised_r, r)

        coarse_loss = self.cd_loss(coarse, gt) 
        fine_loss_1 = self.cd_loss(fine1, gt) 
        fine_loss_2 = self.cd_loss(fine2, gt) 
        fine_loss_3 = self.cd_loss(fine3, gt) 
        return coarse_loss , fine_loss_1 , fine_loss_2 , fine_loss_3, mse_loss

    def forward(self, point_cloud, gt=None):
        #print(point_cloud.shape)
        if self.training and self.pretrain_path and not self.pretrained_loaded:
            from tools.builder import load_model
            load_model(self.base_model, self.pretrain_path)
            print(f"[INFO] Successfully loaded pretrained weights from {self.pretrain_path}")
            self.pretrained_loaded = True  # 确保只加载一次
        pc_xyz = point_cloud.transpose(2,1).contiguous() # point_cloud has shape (B, N, 3) == (B, 2048, 3)

        keypoints_partial_xyz, keyfeatures_partial, _ = self.sa_module_1(pc_xyz, pc_xyz) # keypoints_partial_xyz (B, 3, 512), keyfeatures_partial (B, 128, 512)
        keyfeatures_partial = self.transformer_1(keyfeatures_partial, keypoints_partial_xyz) # (B, in_channel, n) == (B, 128, 512)

        if self.training:
            with torch.no_grad():
                ret, _ = self.base_model.lstnet(
                    point_cloud.transpose(2,1).contiguous(),
                    gt.transpose(2,1).contiguous() if gt is not None else None
                )
            if self.training_mode == 'standard':
                model_output, target = self.Sym_Diffuser(
                    x_original=ret, condition_input=keyfeatures_partial.transpose(2, 1).contiguous(), num_pred=512, dim=12, training_mode='standard'
                )
            elif self.training_mode == 'full_denoise':
                # New training mode: full denoise
                model_output, target, denoised_ret = self.Sym_Diffuser(
                    x_original=ret, condition_input=keyfeatures_partial.transpose(2, 1).contiguous(), num_pred=512, dim=12, training_mode='full_denoise'
                )
            elif self.training_mode == 'proxy_generation':
                model_output, target = self.Sym_Diffuser(
                    x_original=ret, condition_input=keyfeatures_partial.transpose(2, 1).contiguous(), num_pred=512, dim=12, training_mode='standard'
                )
                            
                denoised_ret, proxy_samples_with_indices = self.Sym_Diffuser(
                    x_original=ret, condition_input=keyfeatures_partial.transpose(2, 1).contiguous(), num_pred=512, dim=12, training_mode='proxy_generation',
                    num_proxy_steps=self.num_proxy_steps
                )
                # 🔧 Fix: Use the _build_coarse_geometry method from the DDIM class
                device = point_cloud.device
                proxy_loss = torch.tensor(0.0, device=device)
                total_steps = self.Sym_Diffuser.ddim_num_steps

                for step_idx, proxy_pred in proxy_samples_with_indices:
                    min_weight = 0.1 
                    max_weight = 1.0

                    if total_steps > 1:
                        weight = min_weight + (max_weight - min_weight) * (1.0 - step_idx / (total_steps - 1))
                    else:
                        weight = max_weight # Use max weight if there's only one step
                   
                    coarse_proxy = self._build_coarse_geometry(
                        pred_x0=proxy_pred,
                        keypoints_partial_xyz=keypoints_partial_xyz.transpose(2, 1).contiguous()
                    )

                    proxy_loss += weight * self.cd_loss(coarse_proxy, gt)
            else:
                raise ValueError(f"Unknown training_mode: {self.training_mode}")
        else:
            denoised_ret = self.Sym_Diffuser(
                condition_input=keyfeatures_partial.transpose(2, 1).contiguous(), num_pred=512, dim=12
            )

        b = point_cloud.shape[0]
        R = denoised_ret[:, :, :9].view(b, 512, 3, 3)
        T = denoised_ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints_partial_xyz.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3) # symmetry_points (B, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = self.point_refiner(symmetry_points)
        symmetry_points = symmetry_points.transpose(2, 1).contiguous() # symmetry_points (B, 3, 512)
        

        coarse = torch.cat([symmetry_points, keypoints_partial_xyz], dim=-1) # (B, 3, 512) + (B, 3, 512) == (B, 3, 1024)
       
        feat_symmetry = self.local_encoder(symmetry_points) # feat_symmetry (B, 128, 512)
        feat_partial = keyfeatures_partial

        fine1 = self.MBA_Refiner_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.MBA_Refiner_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)

        fine3 = self.MBA_Refiner_3(
            fine2.transpose(2, 1).contiguous(),  # coarse for the next stage
            feat_symmetry,                         # symmetry_feat
            feat_partial,                          # partial_feat
            symmetry_points.transpose(2, 1).contiguous(),                   # symmetry_xyz
            keypoints_partial_xyz.transpose(2, 1).contiguous()                  # partial_xyz
        )

        rets = [coarse.transpose(2,1).contiguous(), fine1, fine2, fine3] 

        if self.training:
            if self.training_mode == 'proxy_generation':
                return rets, model_output, target, proxy_loss
            else:
                return rets, model_output, target 
        else:
            return rets