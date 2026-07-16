import torch
import torch.nn as nn
import torchvision
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

class Specificity_Estimator(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.conv = FeedForward_MLP(feat_dim, int(feat_dim*0.5))
        
    def forward(self, feat):
        feat = self.conv(feat)
        return feat

class Base_Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self,n_classes):
        super(Base_Model, self).__init__()
        self.fundus_branch = torchvision.models.resnet50(pretrained=True)
        # self.oct_branch = torchvision.models.resnet50(pretrained=True)
        
        self.fundus_branch = nn.Sequential(*list(self.fundus_branch.children())[:-1])  # 去掉 classifier 层
        # for param in self.fundus_branch.parameters():
            # param.requires_grad = False
        # self.oct_branch = nn.Sequential(*list(self.oct_branch.children())[:-1])  # 去掉 classifier 层
        
        #self.global_pool_fundus = nn.AdaptiveAvgPool2d((1, 1))
        #self.global_pool_oct = nn.AdaptiveAvgPool2d((1, 1))
               
        dimension = 2048
        self.specificity_experts = Specificity_Estimator(feat_dim=dimension)
        self.decision_branch = nn.Linear(dimension, n_classes)
        # self.p_missing = p_missing

    def forward(self, fundus_img):
    # def forward(self, inputs):
        # fundus_img, oct_img = inputs
        b1 = self.fundus_branch(fundus_img)
        # b2 = self.oct_branch(oct_img)
        #b1, b2 = self.global_pool_fundus(b1), self.global_pool_oct(b2)
        b1 = torch.flatten(b1, 1)
        fusion_feature = self.specificity_experts(b1)
        logit = self.decision_branch(fusion_feature)

        return logit
