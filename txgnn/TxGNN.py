import os
import math
import argparse
import copy
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import dgl
from dgl.data.utils import save_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from .model import *
from .utils import *

from .TxData import TxData
from .TxEval import TxEval

from .graphmask.moving_average import MovingAverage
from .graphmask.lagrangian_optimization import LagrangianOptimization

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
#device = torch.device("cuda:0")

class TxGNN:
    
    def __init__(self, data,
                       weight_bias_track = False,
                       proj_name = 'TxGNN',
                       exp_name = 'TxGNN',
                       device = 'cuda:0'):
        self.device = torch.device(device)
        self.weight_bias_track = weight_bias_track
        self.G = data.G
        self.df, self.df_train, self.df_valid, self.df_test = data.df, data.df_train, data.df_valid, data.df_test
        self.data_folder = data.data_folder
        self.disease_eval_idx = data.disease_eval_idx
        self.split = data.split
        self.no_kg = data.no_kg
        
        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
        
        self.dd_etypes = [('drug', 'contraindication', 'disease'), 
                  ('drug', 'indication', 'disease'), 
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'), 
                  ('disease', 'rev_indication', 'drug'), 
                  ('disease', 'rev_off-label use', 'drug')]
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        self.config = None
        
    def model_initialize(self, n_hid = 128, 
                               n_inp = 128, 
                               n_out = 128, 
                               proto = True,
                               proto_num = 5,
                               attention = False,
                               sim_measure = 'all_nodes_profile',
                               bert_measure = 'disease_name',
                               agg_measure = 'rarity', 
                               exp_lambda = 0.7,
                               num_walks = 200,
                               walk_mode = 'bit',
                               path_length = 2):
        
        if self.no_kg and proto:
            print('Ablation study on No-KG. No proto learning is used...')
            proto = False
        
        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, n_inp)
        self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(self.df_valid, self.G, 'fix_dst', 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(self.df_test, self.G, 'fix_dst', 1, self.device)

        self.config = {'n_hid': n_hid, 
                       'n_inp': n_inp, 
                       'n_out': n_out, 
                       'proto': proto,
                       'proto_num': proto_num,
                       'attention': attention,
                       'sim_measure': sim_measure,
                       'bert_measure': bert_measure,
                       'agg_measure': agg_measure,
                       'num_walks': num_walks,
                       'walk_mode': walk_mode,
                       'path_length': path_length
                      }

        self.model = HeteroRGCN(self.G,
                   in_size=n_inp,
                   hidden_size=n_hid,
                   out_size=n_out,
                   attention = attention,
                   proto = proto,
                   proto_num = proto_num,
                   sim_measure = sim_measure,
                   bert_measure = bert_measure, 
                   agg_measure = agg_measure,
                   num_walks = num_walks,
                   walk_mode = walk_mode,
                   path_length = path_length,
                   split = self.split,
                   data_folder = self.data_folder,
                   exp_lambda = exp_lambda,
                   device = self.device
                  ).to(self.device)    
        self.best_model = self.model
        
    def pretrain(self, n_epoch = 1, learning_rate = 1e-3, batch_size = 1024, train_print_per_n = 20, sweep_wandb = None):
        
        if self.no_kg:
            raise ValueError('During No-KG ablation, pretraining is infeasible because it is the same as finetuning...')
            
        self.G = self.G.to('cpu')
        print('Creating minibatch pretraining dataloader...')
        train_eid_dict = {etype: self.G.edges(form = 'eid', etype =  etype) for etype in self.G.canonical_etypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.EdgeDataLoader(
            self.G, train_eid_dict, sampler,
            negative_sampler=Minibatch_NegSampler(self.G, 1, 'fix_dst'),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = learning_rate)

        print('Start pre-training with #param: %d' % (get_n_params(self.model)))

        for epoch in range(n_epoch):

            for step, (nodes, pos_g, neg_g, blocks) in enumerate(dataloader):

                blocks = [i.to(self.device) for i in blocks]
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)
                pred_score_pos, pred_score_neg, pos_score, neg_score = self.model.forward_minibatch(pos_g, neg_g, blocks, self.G, mode = 'train', pretrain_mode = True)

                scores = torch.cat((pos_score, neg_score)).reshape(-1,)
                labels = [1] * len(pos_score) + [0] * len(neg_score)

                loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.weight_bias_track:
                    self.wandb.log({"Pretraining Loss": loss})

                if step % train_print_per_n == 0:
                    # pretraining tracking...
                    auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)
                    
                    if self.weight_bias_track:
                        temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Pretraining")
                        temp_d.update({"Pretraining LR": optimizer.param_groups[0]['lr']})
                        self.wandb.log(temp_d)
                    
                    
                    if sweep_wandb is not None:
                        sweep_wandb.log({'pretraining_loss': loss, 
                                  'pretraining_micro_auroc': micro_auroc,
                                  'pretraining_macro_auroc': macro_auroc,
                                  'pretraining_micro_auprc': micro_auprc, 
                                  'pretraining_macro_auprc': macro_auprc})
                    
                    print('Epoch: %d Step: %d LR: %.5f Loss %.4f, Pretrain Micro AUROC %.4f Pretrain Micro AUPRC %.4f Pretrain Macro AUROC %.4f Pretrain Macro AUPRC %.4f' % (
                        epoch,
                        step,
                        optimizer.param_groups[0]['lr'], 
                        loss.item(),
                        micro_auroc,
                        micro_auprc,
                        macro_auroc,
                        macro_auprc
                    ))
        self.best_model = copy.deepcopy(self.model)
        
    def finetune(self, n_epoch = 500, 
                       learning_rate = 1e-3, 
                       train_print_per_n = 5, 
                       valid_per_n = 25,
                       sweep_wandb = None,
                       save_name = None):
        
        best_val_acc = 0

        self.G = self.G.to(self.device)
        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        torch.nn.init.xavier_uniform(self.model.w_rels) # reinitialize decoder
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.8)
        
        for epoch in range(n_epoch):
            negative_graph = neg_sampler(self.G)
            pred_score_pos, pred_score_neg, pos_score, neg_score = self.model(self.G, negative_graph, pretrain_mode = False, mode = 'train')

            pos_score = torch.cat([pred_score_pos[i] for i in self.dd_etypes])
            neg_score = torch.cat([pred_score_neg[i] for i in self.dd_etypes])

            scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
            labels = [1] * len(pos_score) + [0] * len(neg_score)
            loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if self.weight_bias_track:
                self.wandb.log({"Training Loss": loss})

            if epoch % train_print_per_n == 0:
                # training tracking...
                auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)

                if self.weight_bias_track:
                    temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Training")
                    temp_d.update({"LR": optimizer.param_groups[0]['lr']})
                    self.wandb.log(temp_d)

                print('Epoch: %d LR: %.5f Loss %.4f, Train Micro AUROC %.4f Train Micro AUPRC %.4f Train Macro AUROC %.4f Train Macro AUPRC %.4f' % (
                    epoch,
                    optimizer.param_groups[0]['lr'], 
                    loss.item(),
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc
                ))

                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')

            del pred_score_pos, pred_score_neg, scores, labels

            if (epoch) % valid_per_n == 0:
                # validation tracking...
                print('Validation.....')
                (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc), loss = evaluate_fb(self.model, self.g_valid_pos, self.g_valid_neg, self.G, self.dd_etypes, self.device, mode = 'valid')

                if best_val_acc < macro_auroc:
                    best_val_acc = macro_auroc
                    self.best_model = copy.deepcopy(self.model)

                print('Epoch: %d LR: %.5f Validation Loss %.4f,  Validation Micro AUROC %.4f Validation Micro AUPRC %.4f Validation Macro AUROC %.4f Validation Macro AUPRC %.4f (Best Macro AUROC %.4f)' % (
                    epoch,
                    optimizer.param_groups[0]['lr'], 
                    loss,
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc,
                    best_val_acc
                ))

                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')
                
                if sweep_wandb is not None:
                    sweep_wandb.log({'validation_loss': loss, 
                                  'validation_micro_auroc': micro_auroc,
                                  'validation_macro_auroc': macro_auroc,
                                  'validation_micro_auprc': micro_auprc, 
                                  'validation_macro_auprc': macro_auprc})
                
                
                if self.weight_bias_track:
                    temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Validation")
                    temp_d.update({"Validation Loss": loss,
                                  "Validation Relation Performance": self.wandb.Table(data=to_wandb_table(auroc_rel, auprc_rel),
                                        columns = ["rel_id", "Rel", "AUROC", "AUPRC"])
                                  })

                    self.wandb.log(temp_d)
        
        print('Testing...')

        (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc), loss, pred_pos, pred_neg = evaluate_fb(self.best_model, self.g_test_pos, self.g_test_neg, self.G, self.dd_etypes, self.device, True, mode = 'test')

        print('Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
            loss,
            micro_auroc,
            micro_auprc,
            macro_auroc,
            macro_auprc
        ))

        if self.weight_bias_track:
            temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Testing")
            
            temp_d.update({"Testing Loss": loss,
                          "Testing Relation Performance": self.wandb.Table(data=to_wandb_table(auroc_rel, auprc_rel),
                                columns = ["rel_id", "Rel", "AUROC", "AUPRC"])
                          })

            self.wandb.log(temp_d)

        if save_name is not None:
            import pickle
            with open(save_name, 'wb') as f:
                pickle.dump(get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Testing"), f)
            
        print('----- AUROC Performance in Each Relation -----')
        print_dict(auroc_rel, dd_only = False)
        print('----- AUPRC Performance in Each Relation -----')
        print_dict(auprc_rel, dd_only = False)
        print('----------------------------------------------')
        
        
    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
        #save_graphs(os.path.join(path, 'graph_dgl.bin', [self.G]))
    
    def predict(self, df):
        out = {}
        g = self.G
        df_in = df[['x_idx', 'relation', 'y_idx']]
        for etype in g.canonical_etypes:
            try:
                df_temp = df_in[df_in.relation == etype[1]]
            except:
                print(etype[1])
            src = torch.Tensor(df_temp.x_idx.values).to(self.device).to(dtype = torch.int64)
            dst = torch.Tensor(df_temp.y_idx.values).to(self.device).to(dtype = torch.int64)
            out.update({etype: (src, dst)})
        g_eval = dgl.heterograph(out, num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes})
        
        g_eval = g_eval.to(self.device)
        g = g.to(self.device)
        self.model.eval()
        pred_score_pos, pred_score_neg, pos_score, neg_score = self.model(g, 
                                                                           g_eval, 
                                                                           g_eval, 
                                                                           pretrain_mode = False, 
                                                                           mode = 'test')
        return pred_score_pos

    def retrieve_embedding(self, path = None):
        self.G = self.G.to(self.device)
        h = self.model(self.G, self.G, return_h = True)
        for i,j in h.items():
            h[i] = j.detach().cpu()
            
        if path is not None:
            with open(os.path.join(path, 'node_emb.pkl'), 'wb') as f:
                pickle.dump(h, f)
        
        return h
                      
    def retrieve_sim_diseases(self, relation, k = 5, path = None):
        if relation not in ['indication', 'contraindication', 'off-label']:
            raise ValueError("Please select the following three relations: 'indication', 'contraindication', 'off-label' !")
                      
        etypes = self.dd_etypes

        out_degrees = {}
        in_degrees = {}

        for etype in etypes:
            out_degrees[etype] = torch.where(self.G.out_degrees(etype=etype) != 0)
            in_degrees[etype] = torch.where(self.G.in_degrees(etype=etype) != 0)
        
        sim_all_etypes = self.model.pred.sim_all_etypes
        diseaseid2id_etypes = self.model.pred.diseaseid2id_etypes

        id2diseaseid_etypes = {}
        for etype, diseaseid2id in diseaseid2id_etypes.items():
            id2diseaseid_etypes[etype] = {j: i for i, j in diseaseid2id.items()}   
        
        h = self.retrieve_embedding()
        
        if relation == 'indication':
            etype = ('disease', 'rev_indication', 'drug')
        elif relation == 'contraindication':
            etype = ('disease', 'rev_contraindication', 'drug')          
        elif relation == 'off-label':
            etype = ('disease', 'rev_off-label use', 'drug')           
        
        src, dst = etype[0], etype[2]
        src_rel_idx = out_degrees[etype]
        dst_rel_idx = in_degrees[etype]
        src_h = h[src][src_rel_idx]
        dst_h = h[dst][dst_rel_idx]

        src_rel_ids_keys = out_degrees[etype]
        dst_rel_ids_keys = in_degrees[etype]
        src_h_keys = h[src][src_rel_ids_keys]
        dst_h_keys = h[dst][dst_rel_ids_keys]

        h_disease = {}              
        h_disease['disease_query'] = src_h
        h_disease['disease_key'] = src_h_keys
        h_disease['disease_query_id'] = src_rel_idx
        h_disease['disease_key_id'] = src_rel_ids_keys
        
        sim = sim_all_etypes[etype][np.array([diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                      
        ## get top K most similar diseases and their similarity scores
        coef = torch.topk(sim, k + 1).values[:, 1:]
        ## normalize simialrity scores
        coef = F.normalize(coef, p=1, dim=1)
        ## get these diseases embedding
        embed = h_disease['disease_key'][torch.topk(sim, k + 1).indices[:, 1:]]
        ## augmented disease embedding
        out = torch.mul(embed.to('cpu'), coef.unsqueeze(dim = 2)).sum(dim = 1)
        
        similar_diseases = torch.topk(sim, k + 1).indices[:, 1:]
        similar_diseases = similar_diseases.apply_(lambda x: id2diseaseid_etypes[etype][x]) 
        
        if path is not None:
            with open(os.path.join(path, 'sim_diseases.pkl'), 'wb') as f:
                pickle.dump(similar_diseases, f)
                      
        return similar_diseases
                      
    def load_pretrained(self, path):
        ## load config file
        
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        self.model_initialize(**config)
        self.config = config
        #self.G = initialize_node_embedding(self.G, config['n_inp'])
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
        
    def train_graphmask(self, relation = 'indication',
                              learning_rate = 3e-4,
                              allowance = 0.005,
                              epochs_per_layer = 1000,
                              penalty_scaling = 1,
                              moving_average_window_size = 100,
                              valid_per_n = 5):
        
        self.relation = relation
        
        if relation not in ['indication', 'contraindication', 'off-label']:
            raise ValueError("Please select the following three relations: 'indication', 'contraindication', 'off-label' !")
         
        if relation == 'indication':
            etypes_train = [('drug', 'indication', 'disease'),
                            ('disease', 'rev_indication', 'drug')]
        elif relation == 'contraindication':
            etypes_train = [('drug', 'contraindication', 'disease'), 
                           ('disease', 'rev_contraindication', 'drug')]
        elif relation == 'off-label':
            etypes_train = [('drug', 'off-label use', 'disease'),
                           ('disease', 'rev_off-label use', 'drug')]
        else:
            etypes_train = dd_etypes    
        
        best_loss_sum = 100        
        
        if "graphmask_model" not in self.__dict__:
            self.graphmask_model = copy.deepcopy(self.best_model)
            self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            ## add all the parameters for graphmask
            self.graphmask_model.add_graphmask_parameters(self.G)
        else:
            print("Training from checkpoint/pretrained model...")
        
        self.graphmask_model.eval()
        disable_all_gradients(self.graphmask_model)
        
        optimizer = torch.optim.Adam(self.graphmask_model.parameters(), lr=learning_rate)
        self.graphmask_model.to(self.device)
        lagrangian_optimization = LagrangianOptimization(optimizer,
                                                         self.device,
                                                         batch_size_multiplier=None)

        f_moving_average = MovingAverage(window_size=moving_average_window_size)
        g_moving_average = MovingAverage(window_size=moving_average_window_size)

        best_sparsity = 1.01

        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        loss_fct = nn.MSELoss()

        self.G = self.G.to(self.device)
        
        ## iterate over layers. One at a time!
        for layer in reversed(list(range(self.graphmask_model.count_layers()))):
            self.graphmask_model.enable_layer(layer) ## enable baselines and gates parameters

            for epoch in range(epochs_per_layer):
                self.graphmask_model.train()
                neg_graph = neg_sampler(self.G)
                original_predictions_pos, original_predictions_neg, _, _ = self.graphmask_model.graphmask_forward(self.G, self.G, neg_graph, graphmask_mode = False, only_relation = relation)

                pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
                neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
                original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score))).to('cpu')

                updated_predictions_pos, updated_predictions_neg, penalty, num_masked = self.graphmask_model.graphmask_forward(self.G, self.G, neg_graph, graphmask_mode = True, only_relation = relation)
                pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
                neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
                updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

                labels = [1] * len(pos_score) + [0] * len(neg_score)
                loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels).float().to(self.device)).item()

                original_predictions = original_predictions.to(self.device)
                loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels).float().to(self.device)).item()
                # loss is the divergence between updated and original predictions
                loss = loss_fct(original_predictions, updated_predictions)

                g = torch.relu(loss - allowance).mean()
                f = penalty * penalty_scaling

                lagrangian_optimization.update(f, g)

                f_moving_average.register(float(f.item()))
                g_moving_average.register(float(loss.mean().item()))

                print(
                    "Running epoch {0:n} of GraphMask training. Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
                        epoch,
                        g_moving_average.get_value(),
                        f_moving_average.get_value(),
                        loss_pred,
                        loss_pred_ori,
                        num_masked[0]/self.G.number_of_edges(),
                        num_masked[1]/self.G.number_of_edges())
                )

                if self.weight_bias_track == 'True':
                    self.wandb.log({'divergence': g_moving_average.get_value(),
                              'penalty': f_moving_average.get_value(),
                              'bce_masked': loss_pred,
                              'bce_original': loss_pred_ori,
                              '%masked_L1': num_masked[0]/self.G.number_of_edges(),
                              '%masked_L2': num_masked[1]/self.G.number_of_edges()})

                del original_predictions, updated_predictions, f, g, loss, pos_score, neg_score, loss_pred_ori, loss_pred, neg_graph
                
                if epoch % valid_per_n == 0:
                    loss_sum = evaluate_graphmask(self.graphmask_model, self.G, self.g_valid_pos, self.g_valid_neg, relation, epoch, mode = 'validation', allowance = allowance, penalty_scaling = penalty_scaling, etypes_train = etypes_train, device = self.device, weight_bias_track = self.weight_bias_track, wandb = self.wandb)
                    
                    if loss_sum < best_loss_sum:
                        # takes the best checkpoint
                        best_loss_sum = loss_sum
                        self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            
        loss_sum, metrics = evaluate_graphmask(self.best_graphmask_model, self.G, self.g_test_pos, self.g_test_neg, relation, epoch, mode = 'testing', allowance = allowance, penalty_scaling = penalty_scaling, etypes_train = etypes_train, device = self.device, weight_bias_track = self.weight_bias_track, wandb = self.wandb)
        
        if self.weight_bias_track == 'True':
            self.wandb.log(metrics)
        return metrics
    
    def save_graphmask_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_graphmask_model.state_dict(), os.path.join(path, 'graphmask_model.pt'))
        
    def load_pretrained_graphmask(self, path):
        ## load config file
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        self.model_initialize(**config)
        self.config = config
        if "graphmask_model" not in self.__dict__:
            self.graphmask_model = copy.deepcopy(self.best_model)
            self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            ## add all the parameters for graphmask
            self.graphmask_model.add_graphmask_parameters(self.G)
        
        state_dict = torch.load(os.path.join(path, 'graphmask_model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.graphmask_model.load_state_dict(state_dict)
        self.graphmask_model = self.graphmask_model.to(self.device)
        self.best_graphmask_model = self.graphmask_model
    
    
    def retrieve_gates_scores_penalties(self):
        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = self.graphmask_model.graphmask_forward(self.G, self.G, self.G, graphmask_mode = True, only_relation = self.relation, return_gates = True)
        gates = self.graphmask_model.get_gates()
        scores = self.graphmask_model.get_gates_scores()
        penalties = self.graphmask_model.get_gates_penalties()
        
        return gates, scores, penalties
    
    def retrieve_save_gates(self, path):
        _, scores, _ = self.retrieve_gates_scores_penalties()
        
        df_raw = pd.read_csv(os.path.join(self.data_folder, 'kg.csv'))
        df = self.df
        
        df_raw['x_id'] = df_raw.x_id.apply(lambda x: convert2str(x))
        df_raw['y_id'] = df_raw.y_id.apply(lambda x: convert2str(x))

        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

        idx2id_all = {}
        id2name_all = {}
        for node_type in self.G.ntypes:
            idx2id = dict(df[df.x_type == node_type][['x_idx', 'x_id']].values)
            idx2id.update(dict(df[df.y_type == node_type][['y_idx', 'y_id']].values))
            id2name = dict(df_raw[df_raw.x_type == node_type][['x_id', 'x_name']].values)
            id2name.update(dict(df_raw[df_raw.y_type == node_type][['y_id', 'y_name']].values))

            idx2id_all[node_type] = idx2id
            id2name_all[node_type] = id2name
            
        all_att_df = pd.DataFrame()
        
        G = self.G.to('cpu')
        for etypes in G.canonical_etypes:
            etype = etypes[1]
            src, dst = etypes[0], etypes[2]

            df_temp = pd.DataFrame()
            df_temp['x_idx'] = G.edges(etype = etype)[0].numpy()
            df_temp['y_idx'] = G.edges(etype = etype)[1].numpy()
            df_temp['x_id'] = df_temp['x_idx'].apply(lambda x: idx2id_all[src][x])
            df_temp['y_id'] = df_temp['y_idx'].apply(lambda x: idx2id_all[dst][x])

            df_temp['x_name'] = df_temp['x_id'].apply(lambda x: id2name_all[src][x])
            df_temp['y_name'] = df_temp['y_id'].apply(lambda x: id2name_all[dst][x])

            df_temp['x_type'] = src
            df_temp['y_type'] = dst
            df_temp['relation'] = etype

            df_temp[self.relation + '_layer1_att'] = scores[0][etype].reshape(-1,)
            df_temp[self.relation + '_layer2_att'] = scores[1][etype].reshape(-1,)

            all_att_df = all_att_df.append(df_temp)
        
        all_att_df.to_pickle(os.path.join(path, 'graphmask_output_' + self.relation + '.pkl'))
        return all_att_df