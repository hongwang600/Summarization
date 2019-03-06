import torch
CONFIG= {
    'learning_rate': 0.0001,
    'embedding_dim': 100,
    'hidden_dim': 200,
    'batch_size': 32,
    'eval_batch_size': 32,
    'epoch': 300,
    'random_seed': 1,
    'mask_pro': 0.25,
    'loss_margin': 0.5,
    'num_classes': 20,
    'train_file':'data/pretrain/train.txt.src',
    'train_oracle_file':'data/pretrain/train.txt.oracle',
    'train_tgt_file':'data/pretrain/train.txt.tgt',
    'dev_file':'data/pretrain/val.txt.src',
    'dev_oracle_file':'data/pretrain/val.txt.oracle',
    'dev_tgt_file':'data/pretrain/val.txt.tgt',
    'test_file':'data/pretrain/test.txt.src',
    'test_tgt_file':'data/pretrain/test.txt.tgt',
    'model_path':'results/full_aug_data_switch_representation_model.pt',
    'ref_folder': None,
    'pred_folder': None,
    #'summarizer_model_path':'results/summarizer_fintune_model.pt',
    #'summarizer_embed_model_path':'results/representation_model.pt',
    #'summarizer_embed_model_path':None,
    #'load_model_path':'results/sind_best_model_0001.pt',
    #'load_model_path':'results/replace_representation_model.pt',
    'load_model_path':None,
    'exp_name':'runs_pretrain/full_aug_data_switch_model',
    'debug':False,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}
