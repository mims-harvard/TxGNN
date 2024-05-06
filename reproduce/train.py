from txgnn import TxData, TxGNN, TxEval
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--split', type=str, choices = ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative', 'full_graph', 'downstream_pred', 'few_edeges_to_kg'])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default='TxGNN', choices = ['TxGNN', 'GNN', 'TxGAT'])

args = parser.parse_args()

seed = args.seed
TxData = TxData(data_folder_path = './data')
TxData.prepare_split(split = args.split, seed = seed, no_kg = False)


name = '_'.join([args.model, str(args.seed), args.split])
TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN_Baselines',
              exp_name = name,
              device = 'cuda:' + str(args.device)
              )

if args.model == 'GNN':
    proto = False
else:
    proto = True

if args.model == 'TxGAT':
    attention = True
else:
    attention = False
    
TxGNN.model_initialize(n_hid = 100, 
                      n_inp = 100, 
                      n_out = 100, 
                      proto = proto,
                      proto_num = 3,
                      attention = attention,
                      sim_measure = 'all_nodes_profile',
                      agg_measure = 'rarity')

if args.model in ['TxGNN']:

    ## here we did not run this, since the output is too long to fit into the notebook
    TxGNN.pretrain(n_epoch = 1, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)

## here as a demo, the n_epoch is set to 30. Change it to n_epoch = 500 when you use it
TxGNN.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)

TxGNN.save_model('./saved_models/' + name)

from txgnn import TxEval
TxEval = TxEval(model = TxGNN)
result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = False, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False,
                                     save_name = './data/' + name + '_eval')