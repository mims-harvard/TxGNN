from .utils import *

class TxEval:
    
    def __init__(self, model):
        self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model, self.weight_bias_track, self.wandb = model.df, model.df_train, model.df_valid, model.df_test, model.data_folder, model.G, model.best_model, model.weight_bias_track, model.wandb
        self.device = model.device
        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
        self.split = model.split
        
    def eval_disease_centric(self, disease_idxs, relation = None, save_result = False, show_plot = False, verbose = False, save_name = None, return_raw = False, simulate_random = True):
        if self.split == 'full_graph':
            # set only_prediction to True during full graph training
            only_prediction = True
        else:
            only_prediction = False
            
        if disease_idxs == 'test_set':
            disease_idxs = None
        
        self.out = disease_centric_evaluation(self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model,self.device, disease_idxs, relation, self.weight_bias_track, self.wandb, show_plot, verbose, return_raw, simulate_random, only_prediction)
        
        if save_result:
            import pickle, os
            if save_name is None:
                save_name = os.path.join(self.data_folder, 'disease_centric_eval.pkl')
            with open(save_name, 'wb') as f:
                pickle.dump(self.out, f)
        return self.out
    
    def retrieve_disease_idxs_test_set(self, relation):
        relation = 'rev_' + relation
        df_train_valid = pd.concat([self.df_train, self.df_valid])
        df_dd = self.df_test[self.df_test.relation.isin(self.disease_rel_types)]
        df_dd_train = df_train_valid[df_train_valid.relation.isin(self.disease_rel_types)]

        df_rel_dd = df_dd[df_dd.relation == relation]        
        return df_rel_dd.x_idx.unique()
    
    
    def retrieve_all_disease_idxs(self):
        return np.unique(self.df[self.df.x_type == 'disease'].x_idx.unique().tolist() + self.df[self.df.y_type == 'disease'].y_idx.unique().tolist())