import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from dataloader_SPC import generate_dataloader
from utils_SPC import Logger, adjust_learning_rate, CraateLogger,create_cosine_learing_schdule,encode_test_label,set_seed
from model.Two_modality_pretraining_Synergistic import Base_Model
from dependency_SPC import *
from torch import optim
from torchcontrib.optim import SWA
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

#tsne_dir = '/19962387/lijing/mmif-scence-class/image_fusion_moe/MLSDR/constractive_model/'
class_names = ['NV', 'BCC', 'MEL','MISC','SK']


def criterion(logit, truth):

    loss = nn.CrossEntropyLoss()(logit, truth)

    return loss

def metric(logit, truth):
    # prob = F.sigmoid(logit)
    _, prediction = torch.max(logit.data, 1)

    acc = torch.sum(prediction == truth)
    return acc

def train(net,train_dataloader,model_name):

    #net.set_mode('train')
    train_loss = 0
    train_dia_acc = 0  
    train_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()
        
        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
#         meta_data    = meta_data.cuda()
        
        # Diagostic label
        diagnosis_label = label[0].long().cuda()
        # Seven-Point Checklikst labels
#         pn_label = label[1].long().cuda()
#         str_label = label[2].long().cuda()
#         pig_label = label[3].long().cuda()
#         rs_label = label[4].long().cuda()
#         dag_label = label[5].long().cuda()
#         bwv_label = label[6].long().cuda()
#         vs_label = label[7].long().cuda()

        #print()
        logit_diagnosis_fusion,loss_together = net(clinic_image,derm_image)
        #print(logit_diagnosis_fusion.shape)
        
        loss_fusion = criterion(logit_diagnosis_fusion, diagnosis_label)           
        #loss_clic = net.criterion(logit_diagnosis_clic, diagnosis_label)
        #loss_derm = net.criterion(logit_diagnosis_derm, diagnosis_label)
        loss = loss_fusion+loss_together

        dia_acc_fusion = torch.true_divide(metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
        #dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
        #dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

        dia_acc = dia_acc_fusion
        #dia_acc = torch.true_divide(dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

#         sps_acc_fusion = net.metric(logit_pn_fusion, pn_label)
#         sps_acc_clic = net.metric(logit_pn_clic, pn_label)
#         sps_acc_derm = net.metric(logit_pn_derm, pn_label)

#         sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)


        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
#         train_sps_acc += sps_acc.item()

    train_loss = train_loss / (index + 1) # Because the index start with the value 0f zero
    train_dia_acc = train_dia_acc / (index + 1)
#     train_sps_acc = train_sps_acc / (index + 1)

    return train_loss,train_dia_acc

def validation(net,val_dataloader,model_name, epoch):
    net.eval()
    val_loss = 0
    val_dia_acc = 0
    vaL_sps_acc = 0

    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
#         meta_data    = meta_data.cuda()

        diagnosis_label = label[0].long().cuda()
#         pn_label = label[1].long().cuda()
#         str_label = label[2].long().cuda()
#         pig_label = label[3].long().cuda()
#         rs_label = label[4].long().cuda()
#         dag_label = label[5].long().cuda()
#         bwv_label = label[6].long().cuda()
#         vs_label = label[7].long().cuda()

        with torch.no_grad():
          
          
            logits,loss_together = net(clinic_image, derm_image)
            
           
            #batch_intermediate = net.get_intermediate_features(clinic_image, derm_image)
            #batch_attention = net.get_attention_maps(clinic_image, derm_image)
            
            loss = criterion(logits, diagnosis_label)
            loss = loss+loss_together
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits.data, 1)
            
            
            acc = torch.true_divide(metric(logits, diagnosis_label), clinic_image.size(0))
            
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(diagnosis_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            # all_features.extend(features.cpu().numpy())
            
            
            #for k, v in batch_intermediate.items():
                #intermediate_features[k].append(v.cpu().numpy())
            #for k, v in batch_attention.items():
                #attention_maps[k].append(v.cpu().numpy())


          #logit_diagnosis_fusion = net(clinic_image,derm_image)
  
          #loss_fusion = criterion(logit_diagnosis_fusion, diagnosis_label)           
          #loss_clic = net.criterion(logit_diagnosis_clic, diagnosis_label)
          #loss_derm = net.criterion(logit_diagnosis_derm, diagnosis_label)
          #loss = loss_fusion
  
          #dia_acc_fusion = torch.true_divide(metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
          #dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
          #dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))
  
          #dia_acc = dia_acc_fusion
          #dia_acc = torch.true_divide(dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)
  

        val_loss += loss.item()
        val_dia_acc += acc.item()
#         vaL_sps_acc += sps_acc.item()


    
    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    # all_features = np.array(all_features)
    n_classes = all_probs.shape[1]
    
    
    # metrics_dict = calculate_metrics(all_labels, all_preds, all_probs)
    # class_metrics = calculate_class_metrics(all_labels, all_preds, all_probs)
    
    
    #np.save(f'{out_dir}/features.npy', all_features)
    #np.save(f'{out_dir}/labels.npy', all_labels)
    
    # log.write('\nValidation Metrics:\n')
    # for metric_name, value in metrics_dict.items():
    #     log.write(f'{metric_name}: {value:.4f}\n')

    return val_loss,val_dia_acc


def run_train(model_name,mode,i):
    log.write('** start training here! **\n')
    #best_acc = 0
    es = 0
    patience = 50
    best_mean_acc = 0 
    best_loss = 300
    
    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        # train_mode
        train_loss,train_dia_acc = train(net, train_dataloader,model_name)
        log.write('Round: {}, epoch: {}, Train Loss: {:.4f}, Train Dia Acc: {:.4f}\n'.format(i, epoch, train_loss,
                                                                                                         train_dia_acc
                                                                                                         ))

        # validation mode
        val_loss,val_dia_acc = validation(net, val_dataloader,model_name, epoch)
        
        val_acc = val_dia_acc
        val_mean_acc = val_dia_acc
        
        log.write('Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid Dia Acc: {:.4f}\n'.format(i, epoch, val_loss,
                                                                                                         val_dia_acc
                                                                                                         ))

     
        if val_mean_acc > best_mean_acc:
            es = 0
            best_mean_acc = val_mean_acc
            torch.save(net.common_synergistic_encoder.state_dict(), out_dir + '/checkpoint/{synergistic_pretrain}_best_model.pth')
            log.write('Current Best Mean Acc is {}'.format(best_mean_acc))
  
        #if epoch == 150:
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_model.pth')
        if epoch > (epochs - swa_epoch) and epoch % 1 == 0:
            opt.update_swa()
            log.write('SWA Epoch: {}'.format(epoch))

    #torch.save(net, out_dir+'/swa_{}_resnet50_model.pth')

        
if __name__ == '__main__':
    # Hyperparameters
    
    mode = 'multimodal'
    model_name = 'synergistic_SPC_Pretraining'
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = 'Normal'
    deterministic = True
    if deterministic:
        if data_mode == 'Normal':
          random_seeds = 42
        elif data_mode == 'self_evaluated':
          random_seeds = 183
    rounds = 1
    lr = 5e-5
    epochs = 250
    swa_epoch = 50

    train_dataloader, val_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)
    
    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)
      # create logger
        print(random_seeds+i)
        log, out_dir = CraateLogger(mode, model_name,i,data_mode)
        net = Base_Model(n_classes = 5).cuda()
        #net.initialize_memory(train_dataloader)
      # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
      # create learning schdule
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name,mode,i)
