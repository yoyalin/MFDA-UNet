import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from typing import List, Tuple
from nets.efficientNet import EfficientNet

class DWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, padding=1, bias=False, dilation=1, **kwargs):
        super(DWConvLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation,
                                   padding=padding, groups=in_channels, bias=bias,** kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class MultiScaleGroupedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_and_dilations=None):
        super().__init__()
        self.num_branches = len(kernels_and_dilations) if kernels_and_dilations else 4
        self.branch_in_channels = []
        base_channels = in_channels // self.num_branches
        remaining = in_channels % self.num_branches
        for i in range(self.num_branches):
            self.branch_in_channels.append(base_channels + (1 if i < remaining else 0))
        self.branch_out_channels = []
        base_out = out_channels // self.num_branches
        remaining_out = out_channels % self.num_branches
        for i in range(self.num_branches):
            self.branch_out_channels.append(base_out + (1 if i < remaining_out else 0))
        self.branches = nn.ModuleList()
        for i, (kernel_size, dilation_rate) in enumerate(kernels_and_dilations):
            padding = ((kernel_size - 1) * dilation_rate) // 2
            self.branches.append(DWConvLayer(
                self.branch_in_channels[i], self.branch_out_channels[i],
                kernel_size=kernel_size, padding=padding,
                dilation=dilation_rate, bias=False
            ))
    def forward(self, x):
        splits = []
        current = 0
        for ch in self.branch_in_channels:
            splits.append(x[:, current:current+ch, :, :])
            current += ch
        x_out = [branch(split) for branch, split in zip(self.branches, splits)]
        return torch.cat(x_out, dim=1)

class MultiScaleConvFFN(nn.Module):
    def __init__(self, channels, mlp_ratio=4., kernels_and_dilations=None):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.ms_conv = MultiScaleGroupedConv(channels, hidden_channels, kernels_and_dilations)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        return self.proj_out(self.act(self.ms_conv(x)))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_dim = int(embed_dim * qkv_dim_ratio)
        self.head_dim = self.qkv_dim // num_heads
        
        assert self.head_dim >= 32, f"单头维度{self.head_dim}过小，建议减少num_heads或提高qkv_dim_ratio"
        
        self.q_proj = nn.Conv2d(embed_dim, self.qkv_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(embed_dim, self.qkv_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(embed_dim, self.qkv_dim, kernel_size=1, bias=False)
                
        self.out_proj = nn.Conv2d(self.qkv_dim, embed_dim, kernel_size=1, bias=False)

        self.local_conv = nn.Conv2d(
            embed_dim, embed_dim, 
            kernel_size=3, 
            padding=1, 
            groups=embed_dim,
            bias=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        q = self.q_proj(x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.k_proj(x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.v_proj(x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim **-0.5)
        attn = F.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(2, 3).reshape(B, self.qkv_dim, H, W)
        
        attn_out = self.out_proj(attn_out)
        local_out = self.local_conv(x)

        return local_out + attn_out 

class LocalGlobalFusionTransformer(nn.Module):
    def __init__(self, channels, num_heads=4, qkv_dim_ratio=1, mlp_ratio=1):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.mhsa = MultiHeadSelfAttention(
            embed_dim=channels,
            num_heads=num_heads,
            qkv_dim_ratio=qkv_dim_ratio
        )
        
        self.norm2 = LayerNorm2d(channels)
        self.tffn = MultiScaleConvFFN(
            channels,
            mlp_ratio=mlp_ratio,
            kernels_and_dilations=[(3, 1),(3,2),(5,1),(5,2),(1,1)]
        )

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.tffn(self.norm2(x))
        
        return x

class AdaptiveMultiScaleFusionModule(nn.Module):
    def __init__(self, channels_list, final_channels=384):  
        super(AdaptiveMultiScaleFusionModule, self).__init__()
        
        if len(channels_list) == 1:
            self.proj_threshold = channels_list[0]
            threshold_desc = f"single input channel ({channels_list[0]})"
        else:
            sorted_channels = sorted(channels_list, reverse=True)
            self.proj_threshold = sorted_channels[1]
            threshold_desc = f"second largest in {channels_list} (sorted: {sorted_channels})"
        
        print(f"Calculated proj_threshold: {self.proj_threshold} (based on {threshold_desc})")
        
        self.proj_channels_list = []
        for ch in channels_list:
            if ch <= self.proj_threshold:
                proj_ch = ch * 2 if ch * 2 < self.proj_threshold else self.proj_threshold
            else:
                proj_ch = ch
            proj_ch = max(1, proj_ch)
            self.proj_channels_list.append(proj_ch)
        
        print(f"Projection strategy (threshold={self.proj_threshold}):")
        for in_ch, proj_ch in zip(channels_list, self.proj_channels_list):
            action = "keep/expand" if in_ch <= self.proj_threshold else "reduce"
            print(f"  Input {in_ch} → {proj_ch} ({action})")
        
        self.projection_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, proj_ch, 1, bias=False),
                nn.BatchNorm2d(proj_ch)
            ) for in_ch, proj_ch in zip(channels_list, self.proj_channels_list)
        ])
        
        self.fusion_weights = nn.Parameter(torch.ones(len(channels_list)))
        total_proj_channels = sum(self.proj_channels_list)
        self.fusion_conv = nn.Sequential(
            DWConvLayer(total_proj_channels, final_channels, 3, 1, False),
            nn.BatchNorm2d(final_channels), 
            nn.ReLU6(inplace=True)
        )

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]):
        projected_and_resized = []
        for i, feat in enumerate(features):
            proj_feat = self.projection_convs[i](feat)
            resized_feat = F.interpolate(
                proj_feat, size=target_size, mode='bilinear', align_corners=False)
            projected_and_resized.append(resized_feat)
        
        normalized_weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = [feat * normalized_weights[i] for i, feat in enumerate(projected_and_resized)]
        
        fused = torch.cat(weighted_features, dim=1)
        return self.fusion_conv(fused)

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, cbam=None):
        super(unetUp, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.cbam = cbam
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_size)
        )
        self.post_relu = nn.ReLU(inplace=True)
    def forward(self, input_skip, input_up):
        target_h, target_w = input_skip.shape[2:]
        target_size_tensor = torch.tensor((target_h, target_w), device=input_skip.device)
        
        input_up_shape = torch.tensor(input_up.shape[2:], device=input_up.device)
        if not torch.equal(input_up_shape, target_size_tensor):
            input_up_aligned = F.interpolate(
                input_up, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            input_up_aligned = input_up
            
        if self.cbam is not None:
            input_skip = self.cbam(input_skip, input_up_aligned)
            
        outputs = torch.cat([input_skip, input_up_aligned], 1)
        
        conv1_out = self.conv1(outputs)
        conv2_out = self.conv2(conv1_out)
        shortcut_out = self.shortcut(outputs)
        
        if shortcut_out.shape[2:] != conv2_out.shape[2:]:
            shortcut_out = F.interpolate(
                shortcut_out, size=conv2_out.shape[2:], mode='bilinear', align_corners=False
            )
        residual_out = conv2_out + shortcut_out
        residual_out = self.post_relu(residual_out)

        return residual_out
class ChannelAttentionModule(nn.Module):
    """
    通道注意力模块 (CAM)
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return channel_weights

# ---------------------- 1. 添加GSAM模块（用户提供） ----------------------
class GatedSemanticAlignmentModule(nn.Module):
    def __init__(self, high_channels: int, low_channels: int):
        super(GatedSemanticAlignmentModule, self).__init__()
        self.high_channels = high_channels  # 高层特征通道数（解码器上采样后特征）
        self.low_channels = low_channels    # 低层特征通道数（编码器跳跃连接特征）

        # ---------------------- 1. 高层特征处理 ----------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
            )
        
        self.cam = ChannelAttentionModule(low_channels,16)
        self.gate = nn.Sequential(
                nn.Conv2d(low_channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()  # 生成空间级细节权重（1xHxW）
            )
        # ---------------------- 2. 低层特征处理 ----------------------
        self.low_spatial_refiner = nn.Sequential(
            # 1x1 卷积：调整低层特征表达，增强与语义门控的兼容性
            nn.Conv2d(low_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels),
        )


    def forward(self, high_level_feat: torch.Tensor, low_level_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_level_feat: 高层特征（解码器上采样后，高语义、低分辨率）
            low_level_feat: 低层特征（编码器跳跃连接，低语义、高分辨率）
        Returns:
            aligned_low_feat: 语义对齐后的低层特征
        """
        # 1. 高层特征上采样：动态对齐到低层特征的空间尺寸（H, W）
        target_size = low_level_feat.shape[2:]  # 低层特征的目标尺寸 (H, W)
        high_up = F.interpolate(
            high_level_feat,
            size=target_size,
            mode='bilinear',  # 与解码器上采样模式一致
            align_corners=False
        )

        refined_high_origin = self.conv1(high_up)
        
        refined_low = self.low_spatial_refiner(low_level_feat)  # 形状: (B, low_channels, H, W)
        aligned_low = refined_low + refined_high_origin
        refined_high_cam = self.cam(aligned_low) * aligned_low
        
        refined_low_cam = self.gate(refined_high_origin)  # 形状: (B, low_channels, H, W)
        
        aligned = refined_low_cam * refined_high_cam

        return aligned
    
 
class NoOpGSAM(nn.Module):
    """GSAM关闭时的空操作模块"""
    def __init__(self):
        super(NoOpGSAM, self).__init__()

    def forward(self, high_level_feat: torch.Tensor, low_level_feat: torch.Tensor) -> torch.Tensor:
        return low_level_feat

class NoOpFusion(nn.Module):
    """
    带1x1卷积通道缩放的空操作融合模块
    - 当输入通道与目标通道一致时，不执行卷积
    - 当输入尺寸与目标尺寸一致时，不执行插值
    """
    def __init__(self, channels_list, target_channels):
        super(NoOpFusion, self).__init__()
        self.channels_list = channels_list
        self.target_channels = target_channels
        self.highest_ch = channels_list[-1]
        
        if self.highest_ch != self.target_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(self.highest_ch, target_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.ReLU6(inplace=True)
            )
            print(f"[NoOpFusion] Input channel ({self.highest_ch}) != Target ({self.target_channels}). Creating 1x1 Conv.")
        else:
            self.channel_adjust = nn.Identity()
            print(f"[NoOpFusion] Input channel ({self.highest_ch}) == Target ({self.target_channels}). Skipping Conv creation.")
            
        self.proj_channels_list = [target_channels]

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]):
        # 1. 只处理最高层特征
        highest_feat = features[-1]
        
        # 2. 执行通道调整
        adjusted_feat = self.channel_adjust(highest_feat)
        
        current_size = adjusted_feat.shape[2:]
        if current_size == target_size:
            return adjusted_feat
        else:
            resized_feat = F.interpolate(
                adjusted_feat, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            return resized_feat


class Unet(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, is_print_size=False, backbone='efficientnet-b7',
                 fusion_feat_indices: List[int] = [2,3,4],
                 use_gsam=True, use_htm=True, use_multi_scale_fusion=True,** kwargs):
        super(Unet, self).__init__()
        self.is_print_size = is_print_size
        self.backbone_name = backbone
        self.fusion_feat_indices = fusion_feat_indices
        self.use_gsam = use_gsam
        self.use_htm = use_htm
        self.use_multi_scale_fusion = use_multi_scale_fusion
        print(f"use_gsam={self.use_gsam}, use_htm={self.use_htm}, use_multi_scale_fusion={self.use_multi_scale_fusion}")

        if self.backbone_name == 'efficientnet-b5':
            self.encoder_channels = [24, 40, 64, 176, 512]
            weights_file = "efficientnet_b5_86493f6b.pth"
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}. Choose 'b0'/'b4'/'b5'/'b7'.")

        # HTM增强最高层特征（feat5）
        if self.use_htm:
            self.chat_enhancer = LocalGlobalFusionTransformer(
                channels=self.encoder_channels[-1],
                num_heads=4,
                qkv_dim_ratio=1,
                mlp_ratio=2
            )
        else:
            self.chat_enhancer = nn.Identity()
        
        print(f"HTM initialized for feat5: channels={self.encoder_channels[-1]}, num_heads=4")

        
        self._validate_fusion_indices()

        # 加载Backbone
        self.backbone = EfficientNet.from_name(self.backbone_name)
        if pretrained:
            print(f"--- Loading pretrained weights for {self.backbone_name} ---")
            if isinstance(pretrained, bool):
                local_weights_path = os.path.join("model_data", weights_file)
                if os.path.exists(local_weights_path):
                    try:
                        state_dict = torch.load(local_weights_path, weights_only=True)
                        self.backbone.load_state_dict(state_dict)
                        print(f"--- Loaded local weights: {local_weights_path} ---")
                    except Exception as e:
                        print(f"Load local weights failed: {e} → Random init backbone.")
                else:
                    print(f"--- Downloading {self.backbone_name} weights... ---")
                    self.backbone = EfficientNet.from_pretrained(self.backbone_name)
                    print("--- Downloaded weights successfully ---")
            elif isinstance(pretrained, str):
                if os.path.exists(pretrained):
                    try:
                        state_dict = torch.load(pretrained)
                        self.backbone.load_state_dict(state_dict)
                        print(f"--- Loaded custom weights: {pretrained} ---")
                    except Exception as e:
                        print(f"Load custom weights failed: {e} → Random init backbone.")
                else:
                    print(f"Custom weights not found: {pretrained} → Random init backbone.")
            else:
                print("Invalid 'pretrained' argument → Random init backbone.")
        else:
            print(f"--- {self.backbone_name} randomly initialized ---")

        
        self.bottleneck_channels = 512
        self.fusion_channels = [self.encoder_channels[idx] for idx in self.fusion_feat_indices]
        print(f"Fusion input channels: {self.fusion_channels}")
        if self.use_multi_scale_fusion:
            self.bottleneck_fusion = AdaptiveMultiScaleFusionModule(
                channels_list=self.fusion_channels,
                final_channels=self.bottleneck_channels
            )
        else:
            self.bottleneck_fusion = NoOpFusion(
                channels_list=self.fusion_channels,
                target_channels=self.bottleneck_channels
            )
        self._print_fusion_config()

        self.decoder_out_channels = [512, 256, 128, 64]
        self.decoder_up_in_channels = [self.bottleneck_channels] + self.decoder_out_channels[:-1]
        self.decoder_stages = nn.ModuleList()
        self.skip_indices = [3, 2, 1, 0]  
        self.htm_output_channels = self.encoder_channels[-1]  
        
        for i in range(len(self.decoder_out_channels)):
            skip_idx = self.skip_indices[i]
            skip_ch = self.encoder_channels[skip_idx] 
            
            if self.use_gsam:
                if i in [0,1,2]: # 在解码器前3个阶段不使用GSAM
                    high_channels = self.htm_output_channels
                    gsam_module = NoOpGSAM()
                
                else:
                    high_channels = self.decoder_up_in_channels[i] 
                
                    gsam_module = GatedSemanticAlignmentModule(
                        high_channels=high_channels,
                        low_channels=skip_ch
                    )
            else:
                gsam_module = NoOpGSAM()

            self.decoder_stages.append(nn.ModuleDict({
                'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                'gsam': gsam_module,
                'up_concat': unetUp(
                    in_size=skip_ch + self.decoder_up_in_channels[i], 
                    out_size=self.decoder_out_channels[i]
                )
            }))

        self._print_gsam_config()

        # 最终卷积块
        self.final_up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_out_channels[-1], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(16, num_classes, 1)
        print("\n--- Model construction finished ---")

    def _validate_fusion_indices(self):
        valid_indices = list(range(5))
        if not isinstance(self.fusion_feat_indices, list) or len(self.fusion_feat_indices) == 0:
            raise ValueError("'fusion_feat_indices' must be a non-empty list (e.g., [2,3,4]).")
        for idx in self.fusion_feat_indices:
            if idx not in valid_indices:
                raise ValueError(f"Invalid index {idx} in 'fusion_feat_indices'. Must be in {valid_indices}.")
        if len(self.fusion_feat_indices) != len(set(self.fusion_feat_indices)):
            raise ValueError("'fusion_feat_indices' has duplicate indices. Please remove duplicates.")

    def _print_fusion_config(self):
        idx2feat = {0: 'feat1', 1: 'feat2', 2: 'feat3', 3: 'feat4', 4: 'feat5'}
        fusion_feat_names = [idx2feat[idx] for idx in self.fusion_feat_indices]
        print(f"\n--- Multi-Scale Fusion Module Config ---")
        print(f"Fusion Enabled: {self.use_multi_scale_fusion}")
        print(f"Selected Fusion Features: {fusion_feat_names} (feat5 is HTM-enhanced)")
        print(f"Fusion Input Channels: {self.fusion_channels}")
        
        if self.use_multi_scale_fusion:
            print(f"Fusion Output Channel: {self.bottleneck_channels}")
            print(f"Projection Channel List: {self.bottleneck_fusion.proj_channels_list}")
        else:
            print(f"No Fusion: Using 1x1 conv to scale {self.fusion_channels[-1]} → {self.bottleneck_channels}")
            print(f"Projection Channel List: {self.bottleneck_fusion.proj_channels_list}")                                                     

    
    def _print_gsam_config(self):
        idx2feat = {0: 'feat1', 1: 'feat2', 2: 'feat3', 3: 'feat4', 4: 'feat5'}
        print(f"\n--- GSAM Module Config ---")
        for i, stage in enumerate(self.decoder_stages):
            skip_idx = self.skip_indices[i]
            low_ch = self.encoder_channels[skip_idx]
            if i == 0:
                high_source = f"HTM-enhanced feat5 (channels={self.htm_output_channels})"
            else:
                high_source = f"上一阶段输出 (channels={self.decoder_up_in_channels[i]})"
            print(f"Decoder Stage {i+1}: low_channels={low_ch} ({idx2feat[skip_idx]}), high_source={high_source}")

    def visualize_features(self, writer, step, features, gating_signal, htm_attn):
        gate_vis = gating_signal[0, 0].unsqueeze(0)
        writer.add_image("GSAM/Stage1_Gating_Signal", gate_vis, step, dataformats="CHW")
        
        htm_attn_vis = htm_attn[0, 0].unsqueeze(0)
        writer.add_image("HTM/Attention_Map", htm_attn_vis, step, dataformats="CHW")

    
    def forward(self, inputs):
        input_size = inputs.shape[2:]
        input_size_tensor = torch.tensor(input_size, device=inputs.device)
        
        # 编码器提取特征
        endpoints = self.backbone.extract_endpoints(inputs)
        features = [
            endpoints['reduction_1'],  # feat1 (0)
            endpoints['reduction_2'],  # feat2 (1)
            endpoints['reduction_3'],  # feat3 (2)
            endpoints['reduction_4'],  # feat4 (3)
            endpoints['reduction_5'],  # feat5 (4)
        ]

        # HTM增强feat5
        feat5_raw = features[4]
        feat5_size = feat5_raw.shape[2:]
        feat5_size_tensor = torch.tensor(feat5_size, device=feat5_raw.device)
        htm_output = self.chat_enhancer(feat5_raw) 
        
        enhanced_shape = torch.tensor(htm_output.shape[2:], device=htm_output.device)
        if not torch.equal(enhanced_shape, feat5_size_tensor):
            htm_output = F.interpolate(htm_output, size=feat5_size, mode='bilinear', align_corners=False)
        
        features[4] = htm_output 

        if self.is_print_size:
            print(f"\n---- HTM Enhancement for feat5 ----")
            print(f"Raw feat5 shape: {feat5_raw.shape}")
            print(f"HTM output shape (用于第一阶段GSAM): {htm_output.shape}")

        if self.is_print_size:
            print(f"\n---- Encoder Feature Shapes ({self.backbone_name}) ----")
            for i, f in enumerate(features):
                print(f"feat{i+1}: {f.shape}")

        # 多尺度融合
        fusion_inputs = [features[idx] for idx in self.fusion_feat_indices]
        if fusion_inputs:
            fusion_sizes_tensor = torch.stack([
                torch.tensor(feat.shape[2:], device=feat.device, dtype=torch.float32)
                for feat in fusion_inputs
            ])
            areas = fusion_sizes_tensor[:, 0] * fusion_sizes_tensor[:, 1]
            min_idx = torch.argmin(areas)
            target_size = tuple(fusion_sizes_tensor[min_idx].long().cpu().numpy())
        else:
            target_size = (1, 1)
            
        x = self.bottleneck_fusion(fusion_inputs, target_size=target_size)

        if self.is_print_size:
            print(f"\n---- Bottleneck Fusion Details ----")
            idx2feat = {0: 'feat1', 1: 'feat2', 2: 'feat3', 3: 'feat4', 4: 'feat5'}
            for i, (idx, feat) in enumerate(zip(self.fusion_feat_indices, fusion_inputs)):
                feat_note = " (HTM-enhanced)" if idx == 4 else ""
                print(f"Fusion Input {i+1} ({idx2feat[idx]}{feat_note}): {feat.shape}")
            print(f"Fused Bottleneck Shape: {x.shape}")

        if self.is_print_size:
            print(f"\n---- Decoder Process ----")
            print(f"Initial Input (Fused): {x.shape}")

        for i, stage in enumerate(self.decoder_stages):
            skip_idx = self.skip_indices[i]
            raw_skip_feat = features[skip_idx] 
            
            x_up = stage['up'](x)
            
            if i == 0 and self.use_gsam:
                aligned_skip_feat = stage['gsam'](
                    high_level_feat=htm_output,
                    low_level_feat=raw_skip_feat
                )
            else:
                aligned_skip_feat = stage['gsam'](
                    high_level_feat=x_up, 
                    low_level_feat=raw_skip_feat
                )
            
            x = stage['up_concat'](input_skip=aligned_skip_feat, input_up=x_up)
            

        x = self.final_up_conv(x)
        final_output = self.final_conv(x)
        
        final_shape = torch.tensor(final_output.shape[2:], device=final_output.device)
        if not torch.equal(final_shape, input_size_tensor):
            final_output = F.interpolate(final_output, size=input_size, mode='bilinear', align_corners=False)

        if self.is_print_size:
            print(f"\nFinal Output Shape: {final_output.shape}")

        return final_output

