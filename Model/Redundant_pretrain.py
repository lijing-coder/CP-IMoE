import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def loss_pull_together(f1, f2):
        """
        拉近两个特征向量
        f1, f2: [batch, dim]
        return: scalar loss
        """
        # 计算欧式距离
        dist = F.pairwise_distance(f1, f2, p=2)  # [batch]
        loss = torch.mean(dist)  # 最小化
        return loss

class Base_Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self,n_classes):
        super(Base_Model, self).__init__()
        self.fundus_branch = torchvision.models.resnet50(pretrained=False)
        self.oct_branch = torchvision.models.resnet50(pretrained=False)
        
        self.fundus_branch = nn.Sequential(*list(self.fundus_branch.children())[:-1])  # 去掉 classifier 层
        self.fundus_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_1_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{modality_1_backbone_pretrain}_best_model.pth"))
        self.oct_branch = nn.Sequential(*list(self.oct_branch.children())[:-1])  # 去掉 classifier 层
        self.oct_branch.load_state_dict(torch.load("/19962387/lijing/mmif-scence-class/image_fusion_moe/Mixture_of_experts_missing_modality/multimodal_modality_2_Harvard_Pretraining_Normal_weight_file/0/checkpoint/{modality_2_backbone_pretrain}_best_model.pth"))
        for param in self.fundus_branch.parameters():
            param.requires_grad = False
        for param in self.oct_branch.parameters():
            param.requires_grad = False
        
        #self.global_pool_fundus = nn.AdaptiveAvgPool2d((1, 1))
        #self.global_pool_oct = nn.AdaptiveAvgPool2d((1, 1))
               
        dimension = 2048
        self.common_synergistic_encoder = Interaction_Estimator(feat_dim=dimension)
        self.decision_branch = nn.Linear(dimension, n_classes)

        self.w1 = nn.Parameter(torch.tensor(1.0))

    def forward(self, fundus_img, oct_img):
    # def forward(self, inputs):
        # fundus_img, oct_img = inputs
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        #b1, b2 = self.global_pool_fundus(b1), self.global_pool_oct(b2)
        b1, b2 = torch.flatten(b1, 1), torch.flatten(b2, 1)
        fusion_feature = self.common_synergistic_encoder(b1, b2)
        loss_1 = loss_pull_together(b1,fusion_feature)
        loss_2 = loss_pull_together(b2,fusion_feature)
        loss_together = self.w1*(loss_1+loss_2)

        logit = self.decision_branch(fusion_feature)

        return logit,loss_together
