import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def apply_missing_fixed(a, v, p_missing=0.8, device='cuda', mode='alternate', return_mask=False):
    """
    固定顺序的模态缺失模拟（固定前 p% 的样本发生缺失）

    Args:
        a: 模态1张量 [batch, ...] (fundus)
        v: 模态2张量 [batch, ...] (OCT)
        p_missing: 缺失比例 (0~1)，表示前 p% 样本将被置为缺失
        device: 设备
        mode: 缺失模式
              'fundus'    -> 固定前 p% 样本的 fundus 缺失
              'oct'       -> 固定前 p% 样本的 OCT 缺失
              'alternate' -> 固定前 p% 样本交替缺失两种模态
        return_mask: 是否返回mask

    Returns:
        a_new, v_new, (mask_a, mask_v)  # 如果 return_mask=True
        a_new, v_new                    # 如果 return_mask=False
    """
    batch_size = a.size(0)
    num_missing = int(batch_size * p_missing)

    # 初始化mask: 1=缺失, 0=保留
    missing_mask = torch.zeros(batch_size, 2, device=device)

    # 根据模式设定缺失
    if mode == 'fundus':
        missing_mask[:num_missing, 0] = 1  # 前p%缺fundus
    elif mode == 'oct':
        missing_mask[:num_missing, 1] = 1  # 前p%缺OCT
    elif mode == 'alternate':
        for idx in range(num_missing):
            if idx % 2 == 0:
                missing_mask[idx, 0] = 1  # fundus缺失
            else:
                missing_mask[idx, 1] = 1  # OCT缺失
    else:
        raise ValueError("Invalid mode. Choose from ['fundus', 'oct', 'alternate'].")

    # print(missing_mask)
    # 应用缺失
    a_new, v_new = a.clone(), v.clone()
    for i in range(batch_size):
        if missing_mask[i, 0] == 1:
            a_new[i] = torch.zeros_like(a_new[i])
        if missing_mask[i, 1] == 1:
            v_new[i] = torch.zeros_like(v_new[i])

    # 返回
    if return_mask:
        mask_a, mask_v = 1 - missing_mask[:, 0], 1 - missing_mask[:, 1]
        return a_new, v_new, mask_a, mask_v
    else:
        return a_new, v_new


class FeedForward_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super(FeedForward_MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ffn(x))  # 残差 + 归一化

class Interaction_Estimator(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.geno_fc = FeedForward_MLP(feat_dim, int(feat_dim*0.5))
        self.path_fc = FeedForward_MLP(feat_dim, int(feat_dim*0.5))
        self.geno_atten = nn.Linear(feat_dim, 1)
        self.path_atten = nn.Linear(feat_dim, 1)
        
    def forward(self, gfeat, pfeat):        
        g_align = self.geno_fc(gfeat)   # [B, D]
        p_align = self.path_fc(pfeat)   # [B, D]

        inter = g_align * p_align       # [B, D]

        # 注意力从交互中提取，而不是从各自模态单独提取
        geno_att = torch.sigmoid(self.geno_atten(inter))  # [B, 1]
        path_att = torch.sigmoid(self.path_atten(inter))  # [B, 1]

        interaction = geno_att * g_align + path_att * p_align
        return interaction

class Specificity_Estimator(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.conv = FeedForward_MLP(feat_dim, int(feat_dim*0.5))
        
    def forward(self, feat):
        feat = self.conv(feat)
        return feat

class Gating_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, dropout=0.3):
        super(Gating_MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        #self.norm = nn.LayerNorm(output_dim)
        # self.temperature = 3.0

    def forward(self, x):
        out = self.ffn(x)  # 残差 + 归一化
        out = F.softmax(out , dim=-1)  
        # out = F.softmax(out / self.temperature, dim=-1)      # softmax 归一化
        return out

class Base_Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self,n_classes,p_missing,dataset_name):
        super(Base_Model, self).__init__()
        self.fundus_branch = torchvision.models.resnet50(pretrained=False)
        self.oct_branch = torchvision.models.resnet50(pretrained=False)
        
        self.fundus_branch = nn.Sequential(*list(self.fundus_branch.children())[:-1])  # 去掉 classifier 层
        self.oct_branch = nn.Sequential(*list(self.oct_branch.children())[:-1])  # 去掉 classifier 层
        self.common_expert = Interaction_Estimator(feat_dim = 2048)
        self.synergistic_expert = Interaction_Estimator(feat_dim = 2048)
        self.fundus_specfic_expert = Specificity_Estimator(feat_dim=2048)
        self.oct_specfic_expert = Specificity_Estimator(feat_dim=2048)

        if dataset_name == "SPC":
            # print("dataset is SPC")
            self.fundus_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_SPC_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_backbone_pretrain}_best_model.pth"))
            self.oct_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_SPC_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_backbone_pretrain}_best_model.pth"))
            self.common_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_common_SPC_Pretraining_Normal_weight_file/0/checkpoint/{common_pretrain}_best_model.pth"))
            self.synergistic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_synergistic_SPC_Pretraining_Normal_weight_file/0/checkpoint/{synergistic_pretrain}_best_model.pth"))
            self.fundus_specfic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_SPC_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_MLP_pretrain}_best_model.pth"))
            self.oct_specfic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_SPC_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_MLP_pretrain}_best_model.pth"))
        if dataset_name == "MMC":
            # print("dataset is MMC")
            self.fundus_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_MMC-AMD_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_backbone_pretrain}_best_model.pth"))
            self.oct_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_MMC-AMD_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_backbone_pretrain}_best_model.pth"))
            self.common_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_common_MMC_Pretraining_Normal_weight_file/0/checkpoint/{common_pretrain}_best_model.pth"))
            self.synergistic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_synergistic_MMC_Pretraining_Normal_weight_file/0/checkpoint/{synergistic_pretrain}_best_model.pth"))
            self.fundus_specfic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_MMC-AMD_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_MLP_pretrain}_best_model.pth"))
            self.oct_specfic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_MMC-AMD_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_MLP_pretrain}_best_model.pth"))
        if dataset_name == "Harvard":
            # print("dataset is MMC")
            self.fundus_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_backbone_pretrain}_best_model.pth"))
            self.oct_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_backbone_pretrain}_best_model.pth"))
            self.common_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_common_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{common_pretrain}_best_model.pth"))
            self.synergistic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_synergistic_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{synergistic_pretrain}_best_model.pth"))
            self.fundus_specfic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_MLP_pretrain}_best_model.pth"))
            self.oct_specfic_expert.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_MLP_pretrain}_best_model.pth"))

        # 冻结 backbone & experts
        for net in [self.fundus_branch, self.oct_branch,
                    self.common_expert, self.synergistic_expert,
                    self.fundus_specfic_expert, self.oct_specfic_expert]:
            for param in net.parameters():
                param.requires_grad = False

        #类别prompt（不需训练）
        self.complete_prompt = torch.zeros(2048).cuda()  # 初始全 0
        self.missing_fundus_prompt = torch.ones(2048).cuda()  # 初始全 1
        self.missing_oct_prompt = torch.full((2048,), 2.0).cuda()  # 初始全 2

        #特征prompt
        self.prompt_feature_fundus2oct = FeedForward_MLP(dim=2048, hidden_dim=1024)
        self.prompt_feature_oct2fundus = FeedForward_MLP(dim=2048, hidden_dim=1024)

        # gating MLP 输入变大 (8192 维)
        self.Gating_network_MLP = Gating_MLP(dim=4096, hidden_dim=2048, output_dim=4)

        self.decision_branch_forward = FeedForward_MLP(dim=2048, hidden_dim=1024)
        self.decision_branch_linear = nn.Linear(2048, n_classes)
        self.p_missing = p_missing

        # self.weight1 = nn.Parameter(torch.tensor(1.0)) 

    def forward(self, fundus_img, oct_img):
    # def forward(self, inputs):
        # fundus_img, oct_img = inputs
        batch_size = fundus_img.size(0)
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        # print(fundus_img[2,0,22,22])
        
        #b1, b2 = self.global_pool_fundus(b1), self.global_pool_oct(b2)
        b1, b2 = torch.flatten(b1, 1), torch.flatten(b2, 1)
        b1, b2, mask1, mask2 = apply_missing_fixed(b1, b2, p_missing=self.p_missing, mode='oct', return_mask=True)
        # b1, b2, mask1, mask2 = apply_missing(b1, b2, p_missing=self.p_missing, return_mask=True)
        # b1, b2, mask1, mask2 = apply_missing_single(b1, b2, p_missing=self.p_missing, missing_modality="oct", return_mask=True)

        b1_expert_out = self.fundus_specfic_expert(b1)
        b2_expert_out = self.oct_specfic_expert(b2)
        common_expert_out = self.common_expert(b1,b2)
        synergistic_expert_out = self.synergistic_expert(b1,b2)

        prompt_list_type = []
        prompt_list_feature = []
        #Prompt generation        
        for i in range(batch_size):
            if mask1[i] == 1 and mask2[i] == 1:
                # 两个模态都在
                prompt_list_type.append(self.complete_prompt.unsqueeze(0))
                # prompt_list_feature.append(self.complete_prompt.unsqueeze(0))
                prompt_list_feature.append(b1[i]+b2[i].unsqueeze(0))
            elif mask1[i] == 0 and mask2[i] == 1:
                # fundus 缺失
                prompt_list_type.append(self.missing_fundus_prompt.unsqueeze(0))
                prompt_list_feature.append(self.prompt_feature_oct2fundus(b2[i]).unsqueeze(0))
            elif mask2[i] == 0 and mask1[i] == 1:
                # oct 缺失
                prompt_list_type.append(self.missing_oct_prompt.unsqueeze(0))
                prompt_list_feature.append(self.prompt_feature_fundus2oct(b1[i]).unsqueeze(0))

        prompt_class = torch.cat(prompt_list_type, dim=0)
        prompt_feature = torch.cat(prompt_list_feature, dim=0)
        #print(prompt_class.shape,"and",prompt_feature.shape)


        # gating-mlp 输入拼接
        # gating_input = torch.cat([b1, b2, prompt_feature, prompt_class], dim=1)  # [B, 8192]
        gating_input = torch.cat([prompt_feature, prompt_class], dim=1)  # [B, 4096]
        #print(gating_input.shape)
        experts_weight = self.Gating_network_MLP(gating_input)  # [B, 4]
        experts_weight_1, experts_weight_2 = experts_weight[:,0].unsqueeze(1), experts_weight[:,1].unsqueeze(1)
        experts_weight_3, experts_weight_4 = experts_weight[:,2].unsqueeze(1), experts_weight[:,3].unsqueeze(1)

        b1_expert_out = experts_weight_1*b1_expert_out
        b2_expert_out = experts_weight_2*b2_expert_out
        common_expert_out = experts_weight_3*common_expert_out
        synergistic_expert_out = experts_weight_4*synergistic_expert_out
        fusion_feature = b1_expert_out + b2_expert_out + common_expert_out + synergistic_expert_out
        # fusion_feature = torch.cat([b1_expert_out, b2_expert_out, common_expert_out, synergistic_expert_out], dim=1)

        # 加权求和: [batch, dim]
        # fusion_feature = torch.sum(experts_out * experts_weight.unsqueeze(-1), dim=1)
        fusion_feature = self.decision_branch_forward(fusion_feature)
        logit = self.decision_branch_linear(fusion_feature)

        return logit,fusion_feature,experts_weight
