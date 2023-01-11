import scipy.io
import urllib.request
import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import copy
import pickle
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from random import choice
from collections import Counter
import requests
from zipfile import ZipFile 

import warnings
warnings.filterwarnings("ignore")

#device = torch.device("cuda:0")

from .data_splits.datasplit import DataSplitter

def dataverse_download(url, save_path):
    """dataverse download helper with progress bar
    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        print("Local copy not detected... Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()  

def data_download_wrapper(url, save_path):

    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        dataverse_download(url, save_path)
        print("Done!")
        
def preprocess_kg(path, split):
    if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland']:
        
        print('Generating disease area using ontology... might take several minutes...')
        name2id = { 
                    'cell_proliferation': '14566',
                    'mental_health': '150',
                    'cardiovascular': '1287',
                    'anemia': '2355',
                    'adrenal_gland': '9553'
                  }

        ds = DataSplitter(kg_path = path)

        test_kg = ds.get_test_kg_for_disease(name2id[split], test_size = 0.05)
        all_kg = ds.kg
        all_kg['split'] = 'train'
        test_kg['split'] = 'test'
        df = pd.concat([all_kg, test_kg]).drop_duplicates(subset = ['x_index', 'y_index'], keep = 'last').reset_index(drop = True)

        path = os.path.join(path, split + '_kg')
        if not os.path.exists(path):
            os.mkdir(path)

        df.to_csv(os.path.join(path, 'kg.csv'), index = False)
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id', 'split']]

    else:
        ## random, complex disease splits
        df = pd.read_csv(os.path.join(path, 'kg.csv'))
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id']]
    unique_relation = np.unique(df.relation.values)
    undirected_index = []
    
    print('Iterating over relations...')
    
    for i in tqdm(unique_relation):
        if ('_' in i) and (i.split('_')[0] == i.split('_')[1]):
            # homogeneous graph
            df_temp = df[df.relation == i]
            df_temp['check_string'] = df_temp.apply(lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])), axis=1)
            undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist())
        else:
            # undirected
            d_off = df[df.relation == i]
            undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())
    flat_list = [item for sublist in undirected_index for item in sublist]
    df = df[df.index.isin(flat_list)]
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))

    df['x_idx'] = np.nan
    df['y_idx'] = np.nan
    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

    idx_map = {}
    print('Iterating over node types...')
    for i in tqdm(unique_node_types):
        names = np.unique(np.append(df[df.x_type == i]['x_id'].values, df[df.y_type == i]['y_id'].values))
        names2idx = dict(zip(names, list(range(len(names)))))
        df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['x_id'].apply(lambda x: names2idx[x])
        df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['y_id'].apply(lambda x: names2idx[x])
        idx_map[i] = names2idx

    df.to_csv(os.path.join(path, 'kg_directed.csv'), index = False)

def random_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    # to avoid extreme minority types don't exist in valid/test
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)        
        
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}

def disease_eval_fold(df, fold_seed, disease_idx):
    if not isinstance(disease_idx, list):
        disease_idx = np.array([disease_idx])
    else:
        disease_idx = np.array(disease_idx)
        
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    unique_diseases = df_dd.y_idx.unique()
   
    # remove the unique disease out of training
    train_diseases = np.setdiff1d(unique_diseases, disease_idx)
    df_dd_train_val = df_dd[df_dd.y_idx.isin(train_diseases)]                               
    df_dd_test = df_dd[df_dd.y_idx.isin(disease_idx)]
    
    # randomly get 5% disease-drug pairs for validation 
    df_dd_val = df_dd_train_val.sample(frac = 0.05, replace = False, random_state = fold_seed)
    df_dd_train = df_dd_train_val[~df_dd_train_val.index.isin(df_dd_val.index)]
                                       
    df_train = pd.concat([df_not_dd, df_dd_train])
    df_valid = df_dd_val
    df_test = df_dd_test                               
                                   
    #np.random.seed(fold_seed)
    #np.random.shuffle(unique_diseases)
    #train, valid = np.split(unique_diseases, int(0.95*len(unique_diseases)))
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}                      

def complex_disease_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)] 
    
    unique_diseases = df_dd.y_idx.unique()
    np.random.seed(fold_seed)
    np.random.shuffle(unique_diseases)
    train, valid, test = np.split(unique_diseases, [int(frac[0]*len(unique_diseases)), int((frac[0] + frac[1])*len(unique_diseases))])
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test) 
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
    
def create_fold(df, fold_seed = 100, frac = [0.7, 0.1, 0.2], method = 'random', disease_idx = 0.0):
    if method == 'random':
        out = random_fold(df, fold_seed, frac)
    elif method == 'complex_disease':
        out = complex_disease_fold(df, fold_seed, frac)
    elif method == 'downstream_pred':
        out = disease_eval_fold(df, fold_seed, disease_idx)        
    elif method == 'disease_eval':
        out = disease_eval_fold(df, fold_seed, disease_idx)
    elif method == 'full_graph':
        out = random_fold(df, fold_seed, [0.95, 0.05, 0.0])
        out['test'] = out['valid'] # this is to avoid error but we are not using testing set metric here
    else:
        # disease split
        train_val = df[df.split == 'train'].reset_index(drop = True)
        test = df[df.split == 'test'].reset_index(drop = True)
        out = random_fold(train_val, fold_seed, [0.875, 0.125, 0.0])
        out['test'] = test
    return out['train'], out['valid'], out['test']


def create_split(df, split, disease_eval_index, split_data_path, seed):
    df_train, df_valid, df_test = create_fold(df, fold_seed = seed, frac = [0.83125, 0.11875, 0.05], method = split, disease_idx = disease_eval_index)

    unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates()
    df_train = reverse_rel_generation(df, df_train, unique_rel)
    df_valid = reverse_rel_generation(df, df_valid, unique_rel)
    df_test = reverse_rel_generation(df, df_test, unique_rel)
    df_train.to_csv(os.path.join(split_data_path, 'train.csv'), index = False)
    df_valid.to_csv(os.path.join(split_data_path, 'valid.csv'), index = False)
    df_test.to_csv(os.path.join(split_data_path, 'test.csv'), index = False)
    
    return df_train, df_valid, df_test
    
def construct_negative_graph_each_etype(graph, k, etype, method, weights, device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    
    if method == 'corrupt_dst':
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif method == 'corrupt_src':
        neg_dst = dst.repeat_interleave(k)
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
    elif method == 'corrupt_both':
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif (method == 'multinomial_src') or (method == 'inverse_src') or (method == 'fix_src'):
        neg_dst = dst.repeat_interleave(k)
        try:
            neg_src = weights[etype].multinomial(len(neg_dst), replacement=True)
        except:
            neg_src = torch.Tensor([])
    elif (method == 'multinomial_dst') or (method == 'inverse_dst') or (method == 'fix_dst'):
        neg_src = src.repeat_interleave(k)
        try:
            neg_dst = weights[etype].multinomial(len(neg_src), replacement=True)
        except:
            neg_dst = torch.Tensor([])
    return {etype: (neg_src.to(device), neg_dst.to(device))}

def construct_negative_graph(graph, k, device):
    out = {}   
    for etype in graph.canonical_etypes:
        out.update(construct_negative_graph_each_etype(graph, k, etype, device))
    return dgl.heterograph(out, num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

class Minibatch_NegSampler(object):
    def __init__(self, g, k, method):
        if method == 'multinomial_dst':
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'fix_dst':
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict
        
class Full_Graph_NegSampler:
    def __init__(self, g, k, method, device):
        if method == 'multinomial_src':
            self.weights = {
                etype: g.out_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'multinomial_dst':
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'inverse_dst':
            self.weights = {
                etype: -g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'inverse_src':
            self.weights = {
                etype: -g.out_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'fix_dst':
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        elif method == 'fix_src':
            self.weights = {
                etype: (g.out_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        else:
            self.weights = {}
            
        self.k = k
        self.method = method
        self.device = device
    def __call__(self, graph):
        out = {}   
        for etype in graph.canonical_etypes:
            temp = construct_negative_graph_each_etype(graph, self.k, etype, self.method, self.weights, self.device)
            if len(temp[etype][0]) != 0:
                out.update(temp)
            
        return dgl.heterograph(out, num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
        
def evaluate_graph_construct(df_valid, g, neg_sampler, k, device):
    out = {}
    df_in = df_valid[['x_idx', 'relation', 'y_idx']]
    for etype in g.canonical_etypes:
        try:
            df_temp = df_in[df_in.relation == etype[1]]
        except:
            print(etype[1])
        src = torch.Tensor(df_temp.x_idx.values).to(device).to(dtype = torch.int64)
        dst = torch.Tensor(df_temp.y_idx.values).to(device).to(dtype = torch.int64)
        out.update({etype: (src, dst)})
    g_valid = dgl.heterograph(out, num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes})
    
    ng = Full_Graph_NegSampler(g_valid, k, neg_sampler, device)
    g_neg_valid = ng(g_valid)
    return g_valid, g_neg_valid

def get_all_metrics(y, pred, rels):
    edge_dict_ = {v:k for k,v in edge_dict.items()}

    auroc_rel = {}
    auprc_rel = {}
    for rel in np.unique(rels):
        index = np.where(rels == rel)
        y_ = y[index]
        pred_ = pred[index]
        try:
            auroc_rel[edge_dict_[rel]] = roc_auc_score(y_, pred_)
            auprc_rel[edge_dict_[rel]] = average_precision_score(y_, pred_)
        except:
            #print(rel)
            pass
    micro_auroc = roc_auc_score(y, pred)
    micro_auprc = average_precision_score(y, pred)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    
    return auroc_rel, auprc_rel, micro_auroc, \
            micro_auprc, macro_auroc, macro_auprc

def get_all_metrics_fb(pred_score_pos, pred_score_neg, scores, labels, G, full_mode = False):

    auroc_rel = {}
    auprc_rel = {}
    
    if full_mode:
        etypes = G.canonical_etypes
    else:
        etypes = [('drug', 'contraindication', 'disease'), 
                  ('drug', 'indication', 'disease'), 
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'), 
                  ('disease', 'rev_indication', 'drug'), 
                  ('disease', 'rev_off-label use', 'drug')]
        
    for etype in etypes:
        
        try:
            out_pos = pred_score_pos[etype].reshape(-1,).detach().cpu().numpy()
            out_neg = pred_score_neg[etype].reshape(-1,).detach().cpu().numpy()
            pred_ = np.concatenate((out_pos, out_neg))
            y_ = [1]*len(out_pos) + [0]*len(out_neg)
        
            auroc_rel[etype] = roc_auc_score(y_, pred_)
            auprc_rel[etype] = average_precision_score(y_, pred_)
        except:
            pass
    
    micro_auroc = roc_auc_score(labels, scores)
    micro_auprc = average_precision_score(labels, scores)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))

    return auroc_rel, auprc_rel, micro_auroc, \
            micro_auprc, macro_auroc, macro_auprc


def evaluate(model, valid_data, G):
    model.eval()    
    logits_valid, rels = model(G, valid_data) 
    scores = torch.sigmoid(logits_valid) 
    return get_all_metrics(valid_data.label.values, scores.cpu().detach().numpy(), rels)

def evaluate_fb(model, g_pos, g_neg, G, dd_etypes, device, return_embed = False, mode = 'valid'):
    model.eval()
    pred_score_pos, pred_score_neg, pos_score, neg_score = model(G, g_neg, g_pos, pretrain_mode = False, mode = mode)
    
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes])
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
    
    scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
    labels = [1] * len(pos_score) + [0] * len(neg_score)
    loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(device))
            
    if return_embed:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item()
    
def evaluate_graphmask(model, G, g_valid_pos, g_valid_neg, only_relation, epoch, etypes_train, allowance, penalty_scaling, device, mode = 'validation', weight_bias_track = False, wandb = None):
    model.eval()
    G = G.to(device)
    with torch.no_grad():
        loss_fct = nn.MSELoss()

        g_valid_pos = g_valid_pos.to(device)
        g_valid_neg = g_valid_neg.to(device)

        original_predictions_pos, original_predictions_neg, _, _ = model.graphmask_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = False, only_relation = only_relation)

        pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
        original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        original_predictions = original_predictions.to('cpu')

        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = model.graphmask_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = True, only_relation = only_relation)
        pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
        updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        labels = [1] * len(pos_score) + [0] * len(neg_score)
        loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels).float().to(device)).item()

        # loss is the divergence with original predictions
        G = G.to('cpu')
        original_predictions = original_predictions.to(device)
        loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels).float().to(device)).item()

        loss = loss_fct(original_predictions, updated_predictions)

        g = torch.relu(loss - allowance).mean()
        f = penalty * penalty_scaling

        g_valid_pos = g_valid_pos.to('cpu')
        g_valid_neg = g_valid_neg.to('cpu')
        
        print("----- " + mode + " Result -----")
        print("Epoch {0:n}, Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
            epoch,
            float(loss.mean().item()),
            float(f),
            loss_pred,
            loss_pred_ori,
            num_masked[0]/G.number_of_edges(),
            num_masked[1]/G.number_of_edges())
        )
        print("-------------------------------")
        
        if mode == 'testing':
            test_metrics = {}
            pred_update = updated_predictions.detach().cpu().numpy()
            pred_ori = original_predictions.detach().cpu().numpy()
            y_ = np.array(labels)
            
            test_metrics['test auroc original'] = roc_auc_score(y_, pred_ori)
            test_metrics['test auprc original'] = average_precision_score(y_, pred_ori)
            test_metrics['test auroc update'] = roc_auc_score(y_, pred_update)
            test_metrics['test auprc update'] = average_precision_score(y_, pred_update)
            test_metrics['test %masked_L1'] = num_masked[0]/G.number_of_edges()
            test_metrics['test %masked_L2'] = num_masked[1]/G.number_of_edges()
            
        if weight_bias_track:
            wandb.log({mode + ' divergence': float(loss.mean().item()),
                       mode + ' penalty': float(f),
                       mode + ' bce_masked': loss_pred,
                       mode + ' bce_original': loss_pred_ori,
                       mode + ' %masked_L1': num_masked[0]/G.number_of_edges(),
                       mode + ' %masked_L2': num_masked[1]/G.number_of_edges()})

        g_, f_ = float(loss.mean().item()), float(f)
        del original_predictions, updated_predictions, f, g, loss, pos_score, neg_score
    if mode == 'testing':
        return g_ + f_ , test_metrics
    else:
        return g_ + f_
    
def evaluate_mb(model, g_pos, g_neg, G, dd_etypes, device, return_embed = False, mode = 'valid'):
    model.eval()
    #model = model.to('cpu')
    pred_score_pos, pred_score_neg, pos_score, neg_score = model.forward_minibatch(g_pos.to(device), g_neg.to(device), [G.to(device), G.to(device)], G.to(device), mode = mode, pretrain_mode = False)
    
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes])
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
    
    scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
    labels = [1] * len(pos_score) + [0] * len(neg_score)
    loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(device))
    
    model = model.to(device)
    if return_embed:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item()

## disable all gradient
def disable_all_gradients(module):
    for param in module.parameters():
        param.requires_grad = False

def print_dict(x, dd_only = True):
    if dd_only:
        etypes = [('drug', 'contraindication', 'disease'), 
                  ('drug', 'indication', 'disease'), 
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'), 
                  ('disease', 'rev_indication', 'drug'), 
                  ('disease', 'rev_off-label use', 'drug')]
        
        for i in etypes:
            print(str(i) + ': ' + str(x[i]))
    else:
        for i, j in x.items():
            print(str(i) + ': ' + str(j))
        
def to_wandb_table(auroc, auprc):
    return [[idx, i[1], j, auprc[i]] for idx, (i, j) in enumerate(auroc.items())]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def process_df(df_train, edge_dict):
    df_train['relation_idx'] = [edge_dict[i] for i in df_train['relation']]
    df_train = df_train[['x_type', 'x_idx', 'relation_idx', 'y_type', 'y_idx', 'degree', 'label']].rename(columns = {'x_type': 'head_type', 
                                                                                    'x_idx': 'head', 
                                                                                    'relation_idx': 'relation',
                                                                                    'y_type': 'tail_type',
                                                                                    'y_idx': 'tail'})
    df_train['head'] = df_train['head'].astype(int)
    df_train['tail'] = df_train['tail'].astype(int)
    return df_train


def reverse_rel_generation(df, df_valid, unique_rel):
    
    for i in unique_rel.values:
        temp = df_valid[df_valid.relation == i[1]]
        temp = temp.rename(columns={"x_type": "y_type", 
                     "x_id": "y_id", 
                     "x_idx": "y_idx",
                     "y_type": "x_type", 
                     "y_id": "x_id", 
                     "y_idx": "x_idx"})

        if i[0] != i[2]:
            # bi identity
            temp["relation"] = 'rev_' + i[1]
        df_valid = df_valid.append(temp)
    return df_valid.reset_index(drop = True)


def get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, mode):
    
    results = {
              mode + " Micro AUROC": micro_auroc,
              mode + " Micro AUPRC": micro_auprc,
              mode + " Macro AUROC": macro_auroc,
              mode + " Macro AUPRC": macro_auprc
    }
    
    relations = [('drug', 'contraindication', 'disease'),
                 ('drug', 'indication', 'disease'),
                 ('drug', 'off-label use', 'disease'),
                 ('disease', 'rev_contraindication', 'drug'),
                 ('disease', 'rev_indication', 'drug'),
                 ('disease', 'rev_off-label use', 'drug')
                ]
    
    name_mapping = {('drug', 'contraindication', 'disease'): ' Contraindication ',
                    ('drug', 'indication', 'disease'): ' Indication ',
                    ('drug', 'off-label use', 'disease'): ' Off-Label ',
                    ('disease', 'rev_contraindication', 'drug'): ' Rev-Contraindication ',
                    ('disease', 'rev_indication', 'drug'): ' Rev-Indication ',
                    ('disease', 'rev_off-label use', 'drug'): ' Rev-Off-Label '
                   }
    
    for i in relations:
        if i in auroc_rel:
            results.update({mode + name_mapping[i] + "AUROC": auroc_rel[i]})
        if i in auprc_rel:
            results.update({mode + name_mapping[i] + "AUPRC": auprc_rel[i]})
    return results

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def obtain_protein_random_walk_profile(disease, num_walks, path_len, g, disease_etypes, disease_nodes, walk_mode):
    random_walks = []
    num_nodes = len(g.nodes('gene/protein'))
    for _ in range(num_walks):
        successor = g.successors(disease, etype = 'rev_disease_protein')
        if len(successor) > 0:
            current = choice(successor)
        else:
            continue
        path = [current.item()]
        for path_idx in range(path_len):
            successor = g.successors(current, etype = 'protein_protein')
            if len(successor) > 0:
                current = choice(successor)
                path.append(current.item())
            else:
                break

        random_walks = random_walks + path
        
    if walk_mode == 'bit':
        visted_nodes = np.unique(np.array(random_walks))
        node_profile = torch.zeros((num_nodes,))
        node_profile[visted_nodes] = 1.
    elif walk_mode == 'prob':
        visted_nodes = Counter(random_walks)
        node_profile = torch.zeros((num_nodes,))
        for x, y in visted_nodes.items():
            node_profile[x] = y/len(random_walks)
    return node_profile

def obtain_disease_profile(G, disease, disease_etypes, disease_nodes):
    profiles_for_each_disease_types = []
    for idx, disease_etype in enumerate(disease_etypes):
        nodes = G.successors(disease, etype=disease_etype)
        num_nodes = len(G.nodes(disease_nodes[idx]))
        node_profile = torch.zeros((num_nodes,))
        node_profile[nodes] = 1.
        profiles_for_each_disease_types.append(node_profile)
    return torch.cat(profiles_for_each_disease_types)

def exponential(x, lamb):
    return lamb * torch.exp(-lamb * x) + 0.2

def convert2str(x):
    try:
        if '_' in str(x): 
            pass
        else:
            x = float(x)
    except:
        pass

    return str(x)

def map_node_id_2_idx(x, id2idx):
        id_ = convert2str(x)
        if id_ in id2idx:
            return id2idx[id_]
        else:
            return 'null'
        
def process_disease_area_split(data_folder, df, df_test, split):
    disease_file_path = os.path.join(data_folder, 'disease_files')
    disease_list = pd.read_csv(os.path.join(disease_file_path, split + '.csv'))
    
    id2idx = dict(df[df.x_type == 'disease'][['x_id', 'x_idx']].drop_duplicates().values)
    id2idx.update(dict(df[df.y_type == 'disease'][['y_id', 'y_idx']].drop_duplicates().values))

    temp_dict = {}

    # for merged disease ids
    for i,j in id2idx.items():
        try:
            if '_' in i:
                for x in i.split('_'):
                    temp_dict[str(float(x))] = j
        except:
            temp_dict[str(float(i))] = j

    id2idx.update(temp_dict)

    disease_list['node_idx'] = disease_list.node_id.apply(lambda x: map_node_id_2_idx(x, id2idx))

    disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
    temp = df_test[df_test.relation.isin(disease_rel_types)]
    df_test = df_test.drop(temp[~temp.x_idx.isin(disease_list.node_idx.unique())].index)
    
    return df_test


def create_dgl_graph(df_train, df):
    unique_graph = df_train[['x_type', 'relation', 'y_type']].drop_duplicates()
    DGL_input = {}
    for i in unique_graph.values:
        o = df_train[df_train.relation == i[1]][['x_idx', 'y_idx']].values.T
        DGL_input[tuple(i)] = (o[0].astype(int), o[1].astype(int))

    temp = dict(df.groupby('x_type')['x_idx'].max())
    temp2 = dict(df.groupby('y_type')['y_idx'].max())
    temp['effect/phenotype'] = 0.0
    
    output = {}

    for d in (temp, temp2):
        for k, v in d.items():
            output.setdefault(k, float('-inf'))
            output[k] = max(output[k], v)

    g = dgl.heterograph(DGL_input, num_nodes_dict={i: int(output[i])+1 for i in output.keys()})
    
    # get node, edge dictionary mapping relation sent to index
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in g.etypes:
        edge_dict[etype] = len(edge_dict)
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

    return g

def initialize_node_embedding(g, n_inp):
    # initialize embedding xavier uniform
    for ntype in g.ntypes:
        emb = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), n_inp), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data['inp'] = emb
    return g

def disease_centric_evaluation(df, df_train, df_valid, df_test, data_path, G, model, device, disease_ids = None, relation = None, weight_bias_track = False, wandb = None, show_plot = False, verbose = False, return_raw = False, simulate_random = True, only_prediction = False):
    G = G.to(device)
    model = model.eval()
    from sklearn.metrics import accuracy_score, roc_curve, average_precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, f1_score, auc, precision_recall_curve

    dd_etypes = [('drug', 'contraindication', 'disease'), 
               ('drug', 'indication', 'disease'),
               ('drug', 'off-label use', 'disease')]

    dd_rel_types = ['contraindication', 'indication', 'off-label use']

    disease_etypes = [('disease', 'rev_contraindication', 'drug'), 
               ('disease', 'rev_indication', 'drug'),
               ('disease', 'rev_off-label use', 'drug')]

    disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']

    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

    idx2id_drug = dict(df[df.x_type == 'drug'][['x_idx', 'x_id']].drop_duplicates().values)
    idx2id_drug.update(dict(df[df.y_type == 'drug'][['y_idx', 'y_id']].drop_duplicates().values))

    idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
    idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

    df_ = pd.read_csv(os.path.join(data_path, 'kg.csv'))
    df_['x_id'] = df_.x_id.apply(lambda x: convert2str(x))
    df_['y_id'] = df_.y_id.apply(lambda x: convert2str(x))

    id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values)
    id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values))

    id2name_drug = dict(df_[df_.x_type == 'drug'][['x_id', 'x_name']].drop_duplicates().values)
    id2name_drug.update(dict(df_[df_.y_type == 'drug'][['y_id', 'y_name']].drop_duplicates().values))

    drug_ids_rels = {}
    disease_ids_rels = {}

    for i in ['indication', 'contraindication', 'off-label use']:
        drug_ids_rels['rev_' + i] = df[df.relation == i].x_id.unique()
        disease_ids_rels[i] = df[df.relation == i].y_id.unique()

    num_of_drugs_rels = {}
    num_of_diseases_rels = {}
    for i in ['indication', 'contraindication', 'off-label use']:
        num_of_drugs_rels['rev_' + i] = len(drug_ids_rels['rev_' + i])
        num_of_diseases_rels[i] = len(disease_ids_rels[i])


    def mean_reciprocal_rank(rs):
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    def precision_at_k(r, k):
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(r):
        r = np.asarray(r) != 0
        out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    def calculate_metrics(rel, preds_all, labels_all, mode = 'drug', subset_mode = True):
        if mode == 'drug':
            etype = dd_rel_types
            ids_rels = disease_ids_rels
            if subset_mode:
                k10 = int(num_of_diseases_rels[rel] * 0.1)
                k5 = int(num_of_diseases_rels[rel] * 0.05)
                k1 = int(num_of_diseases_rels[rel] * 0.01)
                num_items = num_of_diseases_rels

            else:
                k10 = 2229
                k5 = 1114
                k1 = 222
                num_items = {'indication': 22293, 'contraindication': 22293, 'off-label use': 22293}
        else:
            ids_rels = drug_ids_rels
            if subset_mode:
                k10 = int(num_of_drugs_rels[rel] * 0.1)
                k5 = int(num_of_drugs_rels[rel] * 0.05)
                k1 = int(num_of_drugs_rels[rel] * 0.01)
                num_items = num_of_drugs_rels
            else:
                k10 = 792
                k5 = 396
                k1 = 79
                num_items = {'rev_indication': 7926, 'rev_contraindication': 7926, 'rev_off-label use': 7926}
            etype = disease_rel_types

        if mode == 'drug':
            id2name = id2name_disease
            id2name_rev = id2name_drug
        if mode == 'disease':
            id2name = id2name_drug
            id2name_rev = id2name_disease

        ids_all = list(preds_all[rel].keys())

        name, auroc, auprc =  {}, {}, {}
        acc, sens, spec, f1, ppv, npv, fpr, fnr, fdr, pos_len, ids, ranked_list = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        AP, MRR, Recall, Recall_Random, Enrichment, not_in_ranked_list, in_ranked_list = {}, {}, {}, {}, {}, {}, {}

        disease_not_intersecting_list = []

        k_num = {'1%': k1, '5%': k5, '10%': k10, '10': 10, '50': 50, '100': 100}

        for i, j in k_num.items():
            AP[i], MRR[i], Recall[i], Recall_Random[i], Enrichment[i], not_in_ranked_list[i], in_ranked_list[i] = {}, {}, {}, {}, {}, {}, {}

        for entity_id in ids_all:
            pred = preds_all[rel][entity_id]
            lab = labels_all[rel][entity_id]
            # remove training set drugs/diseases, which are labelled -1
            train_ = [i for i,j in lab.items() if j != -1]
            # retrieving only the drugs/diseases that belong to the rel types
            fixed_keys = np.intersect1d(ids_rels[rel], [i for i,j in lab.items() if j != -1])
            pred_array = np.array([pred[i] for i in fixed_keys])
            lab_array = np.array([lab[i] for i in fixed_keys])

            id2idx = {i: idx for idx, i in enumerate(fixed_keys)}
            idx2id = {idx: i for idx, i in enumerate(fixed_keys)}

            pos_idx = np.where(np.array(lab_array) == 1)[0]
            pos_len[entity_id] = len(pos_idx)
            
            if len(pos_idx) == 0:
                auroc[entity_id] = -1
                auprc[entity_id] = -1
            else:
                try:
                    auroc[entity_id] = roc_auc_score(lab_array, pred_array)
                except:
                    auroc[entity_id] = -1
                try:
                    auprc[entity_id] = average_precision_score(lab_array, pred_array)
                except:    
                    auprc[entity_id] = -1
            
            ranked_list_entity = np.argsort(pred_array)[::-1]
            ranked_list[entity_id] = [id2name[idx2id[i]] for i in ranked_list_entity]
            
            if simulate_random:
                ranked_list_random = []
                for i in range(500):
                    non_guided_drug_list = list(range(len(ranked_list_entity)))
                    np.random.shuffle(non_guided_drug_list)
                    ranked_list_random.append(non_guided_drug_list)
            
            ranked_list_k = {i: ranked_list_entity[:j] for i,j in k_num.items()}       

            for i, j in ranked_list_k.items():
                recalled_list = np.intersect1d(ranked_list_k[i], pos_idx)
                in_ranked_list[i][entity_id] = [id2name[idx2id[x]] for x in recalled_list]
                not_in_ranked_list[i][entity_id] = [id2name[idx2id[x]] for x in pos_idx if x not in recalled_list]
                if len(pos_idx) == 0:
                    Recall[i][entity_id] = -1
                    Recall_Random[i][entity_id] = -1
                    Enrichment[i][entity_id] = -1
                    AP[i][entity_id] = -1
                    MRR[i][entity_id] = -1
                else:
                    Recall[i][entity_id] = len(recalled_list)/len(pos_idx)
                
                    if simulate_random:
                        Recall_Random[i][entity_id] = np.mean([len(np.intersect1d(sim_trial[:k_num[i]], pos_idx))/len(pos_idx) for sim_trial in ranked_list_random])
                    else:
                        Recall_Random[i][entity_id] = k_num[i]/num_items[rel]
                        
                    Enrichment[i][entity_id] = len(recalled_list) / (Recall_Random[i][entity_id] * len(pos_idx))

                    rs = [1 if x in pos_idx else 0 for x in ranked_list_k[i]]
                    AP[i][entity_id] = average_precision(rs)
                    MRR[i][entity_id] = mean_reciprocal_rank([rs])

            y_pred_s = [1 if i else 0 for i in (pred_array >= 0.5)]
            y = lab_array
            cm1 = confusion_matrix(y, y_pred_s)
            if len(cm1) == 1:
                cm1 = np.array([[cm1[0,0], 0], [0, 0]])
            total1=sum(sum(cm1))
            accuracy1=(cm1[0,0]+cm1[1,1])/total1
            acc[entity_id] = accuracy1

            sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            sens[entity_id] = sensitivity1

            specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            spec[entity_id] = specificity1

            f1[entity_id] = f1_score(y, y_pred_s)

            TN = cm1[0][0]
            FN = cm1[1][0]
            TP = cm1[1][1]
            FP = cm1[0][1]

            # Precision or positive predictive value
            ppv[entity_id] = TP/(TP+FP)
            # Negative predictive value
            npv[entity_id] = TN/(TN+FN)
            # Fall out or false positive rate
            fpr[entity_id] = FP/(FP+TN)
            # False negative rate
            fnr[entity_id] = FN/(TP+FN)
            # False discovery rate
            fdr[entity_id] = FP/(TP+FP)
            name[entity_id] = id2name_rev[entity_id]
            ids[entity_id] = entity_id

        out_dict = {'ID': ids,
                'Name': name,
                'Ranked List': ranked_list,            
                'AUROC': auroc, 
                'AUPRC': auprc, 
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'F1': f1,
                'PPV': ppv,
                'NPV': npv,
                'FPR': fpr,
                'FNR': fnr,
                'FDR': fdr,
                '# of Pos': pos_len, 
                'Prediction': preds_all[rel],
                'Labels': labels_all[rel]
               }

        for i in list(k_num.keys()):
            out_dict.update({'Recall@' + i: Recall[i]})
            out_dict.update({'Recall_Random@' + i: Recall_Random[i]})
            out_dict.update({'Enrichment@' + i: Enrichment[i]})
            out_dict.update({'MRR@' + i: MRR[i]})
            out_dict.update({'AP@' + i: AP[i]})
            out_dict.update({'Hits@' + i: in_ranked_list[i]})
            out_dict.update({'Missed@' + i: not_in_ranked_list[i]})

        return out_dict, disease_not_intersecting_list

    def summary(result, rel_type, mode = 'drug', show_plot = True, verbose = True):
        out_dict_mean = {}
        out_dict_std = {}
        for i in list(result.keys()):
            if isinstance(list(result[i].values())[0], (int, float)):
                if verbose:
                    print('---------')
                    print(i + ' mean: ', np.mean(list(result[i].values())))
                    print(i + ' std: ', np.std(list(result[i].values())))
                    print('---------')
                out_dict_mean[i] = np.mean(list(result[i].values()))
                out_dict_std[i] = np.std(list(result[i].values()))

        if show_plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.scatterplot(list(range(len(result['Recall@5%']))), list(result['# of Pos'].values())).set_title("#pos scatter plot")
            plt.show()

            for i in ['Recall@1%', 'Recall@5%', 'Recall@10%', 'Recall@10', 'Recall@50', 'Recall@100', 'AUROC', 'AUPRC', 'MRR@10', 'MRR@50', 'MRR@100', 'AP@10', 'AP@50', 'AP@100']:
                sns.distplot(list(result[i].values())).set_title(i + " distribution")
                plt.show()


            preds_ = np.concatenate([np.array(list(j.values())) for i, j in results['Prediction'].items()]).reshape(-1,)
            labels_ = np.concatenate([np.array(list(j.values())) for i, j in results['Labels'].items()]).reshape(-1,)

            preds_pos = preds_[np.where(labels_ == 1)]
            preds_neg = preds_[np.where(labels_ == 0)]

            sns.distplot(preds_neg).set_title("prediction score distribution")
            sns.distplot(preds_pos)
            plt.show()

        return out_dict_mean, out_dict_std

    def get_scores_disease(rel, disease_ids):
        df_train_valid = pd.concat([df_train, df_valid])
        df_dd = df_test[df_test.relation.isin(disease_rel_types)]
        df_dd_train = df_train_valid[df_train_valid.relation.isin(disease_rel_types)]

        df_rel_dd = df_dd[df_dd.relation == rel]
        df_rel_dd_train = df_dd_train[df_dd_train.relation == rel]
        drug_nodes = G.nodes('drug').cpu().numpy()
        if disease_ids is None:
            disease_ids = df_rel_dd.x_idx.unique()
        preds_contra = {}
        labels_contra = {}
        ids_contra = {}

        for disease_id in tqdm(disease_ids):

            candidate_pos = df_rel_dd[df_rel_dd.x_idx == disease_id][['x_idx', 'y_idx']]
            candidate_pos_train = df_rel_dd_train[df_rel_dd_train.x_idx == disease_id]
            drug_pos = candidate_pos.y_idx.values
            drug_pos_train_val = candidate_pos_train.y_idx.values

            labels = {}
            for i in drug_nodes:
                if i in drug_pos:
                    labels[i] = 1
                elif i in drug_pos_train_val:
                    labels[i] = -1
                    # in the training set
                else:
                    labels[i] = 0

            # construct eval graph
            out = {}
            src = torch.Tensor([disease_id] * len(labels)).to(device).to(dtype = torch.int64)
            dst = torch.Tensor(list(labels.keys())).to(device).to(dtype = torch.int64)
            out.update({('disease', rel, 'drug'): (src, dst)})

            g_eval = dgl.heterograph(out, num_nodes_dict={ntype: G.number_of_nodes(ntype) for ntype in G.ntypes}).to(device)
            
            model.eval()
            _, pred_score_rel, _, pred_score = model(G, g_eval)
            pred = pred_score_rel[('disease', rel, 'drug')].reshape(-1,).detach().cpu().numpy()
            lab = {idx2id_drug[i]: labels[i] for i in g_eval.edges()[1].detach().cpu().numpy()}
            preds_contra[idx2id_disease[disease_id]] = {idx2id_drug[i]: pred[idx] for idx, i in enumerate(g_eval.edges()[1].detach().cpu().numpy())}
            labels_contra[idx2id_disease[disease_id]] = lab
            ids_contra[idx2id_disease[disease_id]] = g_eval.edges()[1].detach().cpu().numpy()

            del pred_score_rel, pred_score
        return preds_contra, labels_contra, drug_nodes, [id2name_drug[idx2id_drug[i]] for i in drug_nodes]
    
    if disease_ids is None:
        # downstream evaluate all test set diseases
        
        temp_d, preds_all, labels_all, org_out_all, metrics_all = {}, {}, {}, {}, {}

        for rel_type in disease_rel_types:
            print('Evaluating relation: ' + rel_type[4:])
            preds_, labels_, drug_idxs, drug_names = get_scores_disease(rel_type, disease_ids)
            preds_all[rel_type], labels_all[rel_type] = preds_, labels_
            results, _ = calculate_metrics(rel_type, preds_all, labels_all, mode = 'disease')
            out_dict_mean, out_dict_std = summary(results, rel_type, mode = 'disease', show_plot = show_plot, verbose = verbose)
            org_out = [[idx, i, out_dict_mean[i], out_dict_std[i]] for idx, i in enumerate(out_dict_mean.keys())]
            org_out_all[rel_type] = org_out
            metrics_all[rel_type] = results
            
            if weight_bias_track:
                temp_d.update({"disease_centric_evaluation_" + rel_type: wandb.Table(data=org_out,
                                    columns = ["metric_id", "metric", "mean", "std"])
                              })

        if weight_bias_track:
            wandb.log(temp_d)
            
        if return_raw:
            out = {'prediction': preds_all, 'label': labels_all, 'summary': org_out_all, 'result': metrics_all}
            return out
        else:
            return {rel_type: pd.DataFrame.from_dict(metrics_all[rel_type]) for rel_type in disease_rel_types}
    
    else:
        # downstream evaluate a specified list of diseases
        temp_d, preds_all, labels_all, metrics_all = {}, {}, {}, {}
        
        rel_type = 'rev_' + relation
        
        preds_, labels_, drug_idxs, drug_names = get_scores_disease(rel_type, disease_ids)
        preds_all[rel_type], labels_all[rel_type] = preds_, labels_
        if only_prediction:
            for i, j in labels_all[rel_type].items():
                for k, l in j.items():
                    if l == -1:
                        labels_all[rel_type][i][k] = 1
            
        results, _ = calculate_metrics(rel_type, preds_all, labels_all, mode = 'disease')
        metrics_all[rel_type] = results
    
        if return_raw:
            out = {'prediction': preds_all[rel_type], 'label': labels_all[rel_type], 'result': metrics_all[rel_type]}
            return out
        else:
            return pd.DataFrame.from_dict(results)
   