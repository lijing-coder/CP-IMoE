import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from dataloader_SPC import generate_dataloader
from utils_SPC import Logger, adjust_learning_rate, CraateLogger,create_cosine_learing_schdule,encode_test_label,set_seed
from model.prompt_moe_our_pretrain import Base_Model
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

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate overall metrics"""
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    specificity = []
    for cls in range(len(np.unique(y_true))):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true_bin, y_pred_bin).ravel()
        spec = tn / (tn + fp)
        specificity.append(spec)
    specificity = np.mean(specificity)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'kappa': kappa
    }

def calculate_class_metrics(y_true, y_pred, y_prob):
    """Calculate per-class metrics"""
    n_classes = len(np.unique(y_true))
    class_metrics = []
    
    for cls in range(n_classes):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        y_prob_bin = y_prob[:, cls]
        
        tn, fp, fn, tp = metrics.confusion_matrix(y_true_bin, y_pred_bin).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        auc_score = metrics.roc_auc_score(y_true_bin, y_prob_bin)
        
        fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_prob_bin)
        
        class_metrics.append({
            'class': cls,
            'class_name': class_names[cls],
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'auc': auc_score,
            'fpr': fpr,
            'tpr': tpr
        })
    
    return class_metrics

def plot_tsne(features, labels, save_path):
    """Generate and save t-SNE plot"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    #colors = ['r','g', 'b', 'c', 'm']
    plt.figure(figsize=(10, 8))
    #scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    #plt.legend(fontsize=18)
    #plt.colorbar(scatter)
    #plt.title('t-SNE visualization')
    #plt.savefig(save_path)
    #plt.close()    
    
    colors = ['r','g', 'b', 'c', 'm']
    num_classes = len(class_names)
    for i in range(num_classes):
        idx = np.array(labels) == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                    c=colors[i], label=class_names[i], alpha=0.6)
    #plt.legend(fontsize=18)
    #save_path = os.path.join(tsne_dir, f"tsne_epoch.jpg")
    plt.savefig(save_path)
    plt.close()

def plot_experts_and_acc(experts_weights_history, acc_history, save_path):
    epochs = range(1, len(acc_history) + 1)
    experts_weights_history = torch.tensor(experts_weights_history)  # [epochs, 4]

    plt.figure(figsize=(8, 6))  # 调整图像大小
    
    # 设置较大的字体
    plt.rcParams.update({'font.size': 16})

    # 四个专家曲线
    for i in range(4):
        plt.plot(epochs, experts_weights_history[:, i], label=f'Expert {i+1} Weight', linewidth=2)
    
    # ACC 曲线（黑色虚线，稍粗）
    # plt.plot(epochs, acc_history, label='ACC', linewidth=3, linestyle='--', color='black')

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    #plt.legend(fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
    plt.close()

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
        logit_diagnosis_fusion, features, _ = net(clinic_image,derm_image)
        #print(logit_diagnosis_fusion.shape)
        
        loss_fusion = criterion(logit_diagnosis_fusion, diagnosis_label)           
        #loss_clic = net.criterion(logit_diagnosis_clic, diagnosis_label)
        #loss_derm = net.criterion(logit_diagnosis_derm, diagnosis_label)
        loss = loss_fusion

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

# 在外部定义全局存储
experts_weights_history = []
acc_history = []

def validation(net,val_dataloader,model_name, epoch):
    net.eval()
    val_loss = 0
    val_dia_acc = 0
    vaL_sps_acc = 0

    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    all_experts_weights = []   # 存当前epoch所有batch的专家权重

    
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
          
          
            logits, features, experts_weight = net(clinic_image, derm_image)
            
           
            #batch_intermediate = net.get_intermediate_features(clinic_image, derm_image)
            #batch_attention = net.get_attention_maps(clinic_image, derm_image)
            
            loss = criterion(logits, diagnosis_label)
            loss = loss
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits.data, 1)
            
            
            acc = torch.true_divide(metric(logits, diagnosis_label), clinic_image.size(0))
            
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(diagnosis_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

            # 记录专家权重（取 batch 内平均值，形状 [4]）
            all_experts_weights.append(experts_weight.mean(dim=0).detach().cpu())
  

        val_loss += loss.item()
        val_dia_acc += acc.item()
#         vaL_sps_acc += sps_acc.item()


    
    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    n_classes = all_probs.shape[1]

    # 专家权重绘图
    avg_experts_weights = torch.stack(all_experts_weights).mean(dim=0)  # [4]

    experts_weights_history.append(avg_experts_weights.numpy())
    avg_acc = metrics.accuracy_score(all_labels, all_preds)
    acc_history.append(avg_acc)
    plot_experts_and_acc(experts_weights_history, acc_history, f'{out_dir}/experts_weight_{epoch}.png')
    
    #参数计算（总体➕各类别）
    metrics_dict = calculate_metrics(all_labels, all_preds, all_probs)
    class_metrics = calculate_class_metrics(all_labels, all_preds, all_probs)
    
    
    #np.save(f'{out_dir}/features.npy', all_features)
    #np.save(f'{out_dir}/labels.npy', all_labels)
    
    # Confusion Matrix
    # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=[f'Class {i}' for i in range(n_classes)],
    #             yticklabels=[f'Class {i}' for i in range(n_classes)])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig(f'{out_dir}/confusion_matrix_{epoch}.png')
    # plt.close()
    # np.savetxt(f'{out_dir}/confusion_matrix_{epoch}.txt', cm, fmt='%d')
    
    # Save class metrics
    auc_file = f'{out_dir}/class_metrics_{epoch}.txt'
    with open(auc_file, 'w') as f:
        f.write('Class\tName\tAUC\tSensitivity\tSpecificity\tPrecision\tF1\n')
        for cls_metric in class_metrics:
            f.write(f"{cls_metric['class']}\t"
                    f"{cls_metric['class_name']}\t"
                    f"{cls_metric['auc']:.4f}\t"
                    f"{cls_metric['sensitivity']:.4f}\t"
                    f"{cls_metric['specificity']:.4f}\t"
                    f"{cls_metric['precision']:.4f}\t"
                    f"{cls_metric['f1']:.4f}\n")
    
    # Plot and save ROC curves for each class
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    
    for cls_metric in class_metrics:
        # Save ROC coordinates to file
        roc_file = f"{out_dir}/roc_{cls_metric['class_name']}_{epoch}.txt"
        with open(roc_file, 'w') as f:
            f.write(f"Class {cls_metric['class']} ({cls_metric['class_name']}) ROC Coordinates:\n")
            f.write("FPR,TPR\n")
            for fpr, tpr in zip(cls_metric['fpr'], cls_metric['tpr']):
                f.write(f"{fpr:.6f},{tpr:.6f}\n")
            f.write(f"AUC: {cls_metric['auc']:.6f}\n")
        
        # Plot ROC curve
        plt.plot(cls_metric['fpr'], cls_metric['tpr'], 
                 color=colors[cls_metric['class']], 
                 lw=2, 
                 label=f"{cls_metric['class_name']} (AUC = {cls_metric['auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt.legend(fontsize=20)
    plt.savefig(f'{out_dir}/roc_classes_{epoch}.png')
    plt.close()
    
    # Plot t-SNE
    #print(all_features)
    # plot_tsne(all_features, all_labels, f'{out_dir}/tsne_{epoch}.png')
    
    log.write('\nValidation Metrics:\n')
    for metric_name, value in metrics_dict.items():
        log.write(f'{metric_name}: {value:.4f}\n')

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
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_best_model.pth')
            log.write('Current Best Mean Acc is {}'.format(best_mean_acc))
        #  else:
        #      es += 1
        #      print("Counter {} of {}".format(es,patience))
          
        #      if es > patience:
        #          print("Early stopping with best_mean_acc: {:.4f}".format(best_mean_acc), "and val_mean_acc for this epoch: {:.4f}".format(val_mean_acc))
        #          break
  
        #if epoch == 150:
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_model.pth')
        if epoch > (epochs - swa_epoch) and epoch % 1 == 0:
            opt.update_swa()
            log.write('SWA Epoch: {}'.format(epoch))

    #torch.save(net, out_dir+'/swa_{}_resnet50_model.pth')

        
if __name__ == '__main__':
    # Hyperparameters
    
    mode = 'multimodal'
    model_name = 'our-SPC_missing=0.4_random_missing'
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
        net = Base_Model(n_classes = 5,p_missing=0.4,dataset_name="SPC").cuda()
        #net.initialize_memory(train_dataloader)
      # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
      # create learning schdule
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name,mode,i)
