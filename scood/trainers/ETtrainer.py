import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scood.losses import rew_ce, rew_sce,Extension
from scood.utils import sort_array
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer


class ETtrainer(BaseTrainer):
    def __init__(
            self,
            net: nn.Module,
            labeled_train_loader: DataLoader,
            unlabeled_train_loader: DataLoader,
            labeled_aug_loader: DataLoader,
            unlabeled_aug_loader: DataLoader,
            lamda: float,
            learning_rate: float = 0.1,
            min_lr: float = 1e-6,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            epochs: int = 100,
            num_clusters: int = 1000,
            t: float = 0.5,
            lambda_oe: float = 0.5,
            lambda_rep: float = 0.3,
    ) -> None:
        super().__init__(
            net,
            labeled_train_loader,
            learning_rate=learning_rate,
            min_lr=min_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
        )

        # self.epochs = epochs
        self.unlabeled_train_loader = unlabeled_train_loader
        self.labeled_aug_loader = labeled_aug_loader
        self.unlabeled_aug_loader = unlabeled_aug_loader
        self.lamda = lamda

        self.num_clusters = num_clusters
        self.t = t
        self.lambda_oe = lambda_oe
        self.lambda_rep = lambda_rep
        self.hc = 1
        self.K = 128
        self.outs = [self.K] * self.hc


    def train_epoch(self, epoch):
        if epoch >  - 1:
            self._run_clustering(epoch)
        metrics = self._compute_loss(epoch)

        return metrics

    def _compute_loss(self, epoch):
        self.net.train()  # enter train mode
        loss_avg = 0.0
        train_dataiter = iter(self.labeled_aug_loader)
        unlabeled_dataiter = iter(self.unlabeled_aug_loader)

        criterion_rep = Extension(temperature=self.t, scale_by_temperature=False)
        for train_step in range(1, len(train_dataiter) + 1):  
            batch = next(train_dataiter) 
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.unlabeled_aug_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            labeled_aug_data = batch["aug_data"]
            q_labeled_aug_data = labeled_aug_data[0].cuda()
            unlabeled_aug_data = unlabeled_batch["aug_data"]
            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data])
            N1 = len(q_labeled_aug_data)
            con_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True)  
            concat_label = torch.cat([batch["label"], unlabeled_batch["pseudo_label"].type_as(
                batch["label"]), ]) 
            loss_rep = criterion_rep(con_logits_rep, concat_label.cuda())
            logits_augcls, logits_oe_augcls = con_logits_cls[:N1], con_logits_cls[N1:]
            cluster_ID_label = unlabeled_batch["pseudo_label"]
            cluster_ID_label = cluster_ID_label.type_as(batch["label"])
            '''standard CE loss(labeled ID+cluster ID)'''
            loss_cls = F.cross_entropy(con_logits_cls[concat_label != -1] / 0.5, concat_label[
                concat_label != -1].cuda(), ) + 0.3 * F.cross_entropy(logits_augcls / 0.5, batch["label"].cuda(), ) 
            # oe loss
            concat_softlabel = torch.cat([batch["soft_label"], unlabeled_batch["pseudo_softlabel"]])
            loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )

            loss = loss_cls + loss_rep + 0.5 * loss_oe

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.scheduler.step()  
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics["train_loss"] = loss_avg
        return metrics
   
    def EOT(self, N, OT_logits_list, epoch):
        sam_energy = torch.logsumexp(OT_logits_list / 1000, dim=1).cuda()
        # print('sam_energy_std = ', torch.std(sam_energy))
        PS = nn.functional.softmax(OT_logits_list, 1).cuda()
        # if epoch % 50 == 0:
        #     N, K = OT_logits_list.shape
        #     print('sam_std=', sam_energy.reshape((1, N)), flush=True)
        PS = PS.detach()
        N, K = PS.shape
        tt = time.time()
        PS = PS.T  # now it is K x N
        r = (torch.ones((K, 1)) / K).double()
        # r = F.normalize(r, p=1, dim=0)
        r = r.cuda()
        c = (torch.ones((N, 1)) / N).double()
        # c = F.normalize(c, p=1, dim=0)
        c = c.cuda()
        PS = torch.pow(PS, self.lamda).double()  # K x N
        inv_K = (1. / K)
        inv_K = (torch.tensor(inv_K)).double()
        inv_K = inv_K.cuda()
        inv_N = (sam_energy.reshape((N, 1))).double()
        inv_N = F.normalize(inv_N, p=1, dim=0)
        inv_N = inv_N.cuda()
        err = 1e3
        step = 0
        while err > 1e-1:
            r = inv_K / (PS @ c)  # (KxN)@(N,1) = K x 1
            c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
            if step % 5 == 0:
                sin_err = torch.abs(c / c_new - 1)
                sin_err = torch.where(torch.isnan(sin_err), torch.full_like(sin_err, 0), sin_err)
                err = torch.sum(sin_err)
            c = c_new
            step += 1
        PS *= np.squeeze(c) 
        PS = PS.T  # N × K
        PS *= np.squeeze(r)  
        PS = PS.T  # K × N
        PS = torch.where(torch.isnan(PS), torch.full_like(PS, 0), PS)
        label_ET = torch.argmax(PS, 0)  # size N
        print('step =', step, "err =", err.item(), flush=True)
        return label_ET

    def _run_clustering(self, epoch):
        self.net.train()
        start_time = time.time()

        # get data from train loader
        print(
            "######### ET: gathering OT_logits... ############",
            flush=True,  
        )
        train_idx_list, unlabeled_idx_list, OT_logits_list, train_label_list = ([],[],[],[],)
        train_dataiter = iter(self.labeled_train_loader)
        for step in range(1, len(train_dataiter) + 1):  
            batch = next(train_dataiter)  
            index = batch["index"] 
            label = batch["label"]  
            # we use no augmented image for clustering
            data = batch["OT_data"].cuda()  # "plain_data" is basic image transformation for online clustering (without augmentations)
            _, OT_logits = self.net(data, return_aux=True) 
            OT_logits = OT_logits.detach()  
            train_idx_list.append(index)
            train_label_list.append(label)
            OT_logits_list.append(OT_logits)  

        train_idx_list = torch.cat(train_idx_list)
        train_label_list = torch.cat(train_label_list)
        OT_logits_list = torch.cat(OT_logits_list, dim=0).cpu().tolist()
        num_train_data = len(OT_logits_list)  
        train_idx_list = np.array(train_idx_list, dtype=int) 
        train_label_list = np.array(train_label_list, dtype=int)
       

        train_label_list = sort_array(train_label_list, train_idx_list)  
        # in-distribution samples always have pseudo labels == actual labels
        self.labeled_train_loader.dataset.pseudo_label = train_label_list

        torch.cuda.empty_cache() 

        # gather unlabeled image feature in order
        unlabeled_conf_list, unlabeled_pseudo_list, OT_oe_logits_list, unlabeled_sc_list, out_energy_list = [], [], [], [], []
        unlabeled_dataiter = iter(self.unlabeled_train_loader)
        for step in range(1, len(unlabeled_dataiter) + 1):  
            batch = next(unlabeled_dataiter)
            index = batch["index"]
            # we use no augmented image for clustering
            data = batch["OT_data"].cuda()  
            sclabel = batch["sc_label"]
            logit, OT_logits = self.net(data, return_aux=True) 
            OT_logits = OT_logits.detach() 
            logit = logit.detach()  
            score = torch.softmax(logit, dim=1)  
            conf, pseudo = torch.max(score, dim=1)  
           
            unlabeled_idx_list.append(index)
            unlabeled_conf_list.append(conf)
            OT_oe_logits_list.append(OT_logits)
            unlabeled_sc_list.append(sclabel)

        OT_oe_logits_list = torch.cat(OT_oe_logits_list, dim=0).cpu().tolist()
        unlabeled_idx_list = torch.cat(unlabeled_idx_list)
        unlabeled_conf_list = torch.cat(unlabeled_conf_list).cpu()
        unlabeled_sc_list = torch.cat(unlabeled_sc_list)
        OT_logits_list.extend(OT_oe_logits_list)
        OT_logits_list = torch.tensor(OT_logits_list).cuda()

        unlabeled_idx_list = np.array(unlabeled_idx_list, dtype=int)
        unlabeled_conf_list = np.array(unlabeled_conf_list)
        # unlabeled_pseudo_list = np.array(unlabeled_pseudo_list)
        unlabeled_sc_list = np.array(unlabeled_sc_list)

        unlabeled_conf_list = sort_array(unlabeled_conf_list,
                                         unlabeled_idx_list) 
        # unlabeled_pseudo_list = sort_array(unlabeled_pseudo_list,unlabeled_idx_list) 
        unlabeled_sc_list = sort_array(unlabeled_sc_list, unlabeled_idx_list)
        torch.cuda.empty_cache()  

        N = len(OT_logits_list)  # 150000

        with torch.no_grad():  
            label_ET = self.EOT(N, OT_logits_list, epoch)

        label_ET = label_ET.tolist()
        train_cluster_id = label_ET[:num_train_data] 

        unlabeled_cluster_id = label_ET[num_train_data:]  
        # assign cluster id to samples. Sorted by shuffle-recording index.
        train_cluster_id = sort_array(train_cluster_id, train_idx_list)
        unlabeled_cluster_id = sort_array(unlabeled_cluster_id, unlabeled_idx_list)
        self.labeled_train_loader.dataset.cluster_id = train_cluster_id
        self.unlabeled_train_loader.dataset.cluster_id = unlabeled_cluster_id  
        cluster_id = np.concatenate([train_cluster_id, unlabeled_cluster_id]) 
        cluster_stat = np.zeros(self.num_clusters)
        cluster_id_list, cluster_id_counts = np.unique(cluster_id,
                                                       return_counts=True)  
        for cluster_idx, counts in zip(cluster_id_list, cluster_id_counts):  
            cluster_stat[cluster_idx] = counts  

        old_train_pseudo_label = self.labeled_train_loader.dataset.pseudo_label 
        old_unlabeled_pseudo_label = self.unlabeled_train_loader.dataset.pseudo_label 
        old_pseudo_label = np.append(old_train_pseudo_label, old_unlabeled_pseudo_label).astype(int)
        new_pseudo_label = (-1 * np.ones_like(old_pseudo_label)).astype(int)  
        for cluster_idx in range(self.num_clusters): 
            label_in_cluster, label_counts = np.unique(old_pseudo_label[cluster_id == cluster_idx],
                                                       return_counts=True)
            cluster_size = len(
                old_pseudo_label[cluster_id == cluster_idx])  
            purity = label_counts / cluster_size  
            # if purity.size == 0:
            #     maxpurity = 0
            #     plindex = -1
            # else:
            #     maxpurity = np.max(purity)
            #     plindex = np.argmax(purity)

            if np.any(purity > 0.5):
                majority_label = label_in_cluster[purity > 0.5][0]
                new_pseudo_label[cluster_id == cluster_idx] = majority_label  

        self.unlabeled_train_loader.dataset.pseudo_label = new_pseudo_label[num_train_data:] 

        print("######### Label Assignment Completed! Duration: {:.2f}s ############".format(time.time() - start_time),flush=True, )
