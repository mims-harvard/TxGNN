import numpy as np
import pandas as pd
import torch
from .do_obo_parser import OBOReader as DO_Reader
import os
dirname = os.path.dirname(__file__)

class DataSplitter: 
    
    def __init__(self, kg_path=''): 
        self.kg, self.nodes, self.edges = self.load_kg(kg_path)
        self.edge_index = torch.LongTensor(self.edges.get(['x_index', 'y_index']).values.T)
        self.doid2parent, self.doid2name, self.doid2children = self.load_do()
        self.mondo_xref = pd.read_csv(os.path.join(dirname, 'mondo_references.csv'))
        #self.grouped_diseases = pd.read_csv('kg_grouped_diseases.csv')
        self.grouped_diseases = pd.read_csv(os.path.join(dirname, 'kg_grouped_diseases_bert_map.csv'))
        
    def load_kg(self, pth=''):
        kg = pd.read_csv(pth+'kg.csv', low_memory=False)
        nodes = pd.read_csv(pth+'nodes.csv', low_memory=False)
        edges = pd.read_csv(pth+'edges.csv', low_memory=False)
        return kg, nodes, edges
    
    def load_do(self): 
        data = [*iter(DO_Reader(os.path.join(dirname, 'HumanDO.obo')))]
        doid2parent = {}
        for x in data: 
            for parent in x._parents: 
                doid2parent[x.item_id] = parent
        doid2name = {}
        for x in data: 
            doid2name[x.item_id] = x.name
        doid2children = {}
        for x in data: 
            for parent in x._parents:
                if parent not in doid2children: 
                    doid2children[parent] = set()
                doid2children[parent].add(x.item_id)
        for depth in range(20): 
            for parent, children in doid2children.items():
                new_children = set()
                for child in children: 
                    if child in doid2children: 
                        grandkids = doid2children[child]
                        for kid in grandkids:
                            new_children.add(kid)
                doid2children[parent] = doid2children[parent].union(new_children)
        return doid2parent, doid2name, doid2children
    
    def get_nodes_for_doid(self, code): 
        doids = self.doid2children[code].copy()
        doids.add(code)
        mondo = self.mondo_xref.query('ontology == "DOID" and ontology_id in @doids').get(['mondo_id']).drop_duplicates().values.reshape(-1).astype('str')   
        idx1 = self.nodes.query('node_id in @mondo and node_source == "MONDO"').get('node_index').values
        mondo_grp = self.grouped_diseases.query('node_id in @mondo and node_source == "MONDO"').get(['group_id_bert']).drop_duplicates().values.reshape(-1).astype('str')   
        idx2 = self.nodes.query('node_id in @mondo_grp and node_source == "MONDO_grouped"').get('node_index').values
        node_idx = np.unique(np.concatenate([idx1, idx2]))
        return node_idx
        
    def get_nodes_df_for_diod(self, code): 
        node_idx = self.get_nodes_for_doid(code)
        df = self.nodes.query('node_index in @node_idx')
        return df
    
    def get_edge_group(self, nodes, test_size = 0.05, add_drug_dis=True): 
        test_num_edges = round(self.edge_index.shape[1]*test_size)
        
        if add_drug_dis: 
            x = self.edges.query('x_index in @nodes or y_index in @nodes').query('relation=="contraindication" or relation=="indication" or relation=="off-label use"')
            drug_dis_edges = x.get(['x_index','y_index']).values.T
            num_random_edges = test_num_edges - drug_dis_edges.shape[1]
        else: 
            num_random_edges = test_num_edges
            
        from torch_geometric.utils import k_hop_subgraph
        subgraph_nodes, filtered_edge_index, node_map, edge_mask = k_hop_subgraph(list(nodes), 2, self.edge_index) #one hop neighborhood
        sample_idx = np.random.choice(filtered_edge_index.shape[1], num_random_edges, replace=False)
        sample_edges = filtered_edge_index[:, sample_idx].numpy()
        
        if add_drug_dis:
            test_edges = np.concatenate([drug_dis_edges, sample_edges], axis=1)
        else: 
            test_edges = sample_edges
        
        test_edges = np.unique(test_edges, axis=1)
        return test_edges 
        
    def get_test_kg_for_disease(self, doid_code, test_size = 0.05, add_drug_dis=True): 
        disease_nodes = self.get_nodes_for_doid(doid_code)
        disease_edges = self.get_edge_group(disease_nodes, test_size = test_size, add_drug_dis=add_drug_dis)
        disease_edges = pd.DataFrame(disease_edges.T, columns=['x_index','y_index'])
        select_kg = pd.merge(self.kg, disease_edges, 'right').drop_duplicates()
        return select_kg
    
    
    
'''
Usage

ds = DataSplitter(kg_path='../knowledge_graph/kg_giant.csv')
test_kg = ds.get_test_kg_for_disease('14566')
'''

'''
Diseases selected for testing

    Code     Name

    14566    cell proliferation
    150      mental health
    1287     cardiovascular system disease
    2355     anemia
    9553     adrenal gland disease
'''    