import torch
CONFIG= {
    'learning_rate': 0.0001,
    'embedding_dim': 100,
    'hidden_dim': 200,
    'batch_size': 32,
    'eval_batch_size': 32,
    'epoch': 300,
    'random_seed': 1,
    'mask_pro': 0.15,
    'loss_margin': 0.5,
    'train_file':'data/summarization/train.txt.src',
    'train_oracle_file':'data/summarization/train.txt.oracle',
    'dev_file':'data/summarization/val.txt.src',
    'dev_oracle_file':'data/summarization/val.txt.oracle',
    'dev_tgt_text_file':'data/summarization/val.txt.tgt',
    'test_file':'data/summarization/test.txt.src',
    'test_tgt_text_file':'data/summarization/test.txt.tgt',
    'model_path':'results/representation_model_10_layer.pt',
    'summarizer_model_path':'results/summarizer_fintune_model.pt',
    'summarizer_embed_model_path':'results/representation_model.pt',
    #'summarizer_embed_model_path':None,
    #'load_model_path':'results/sind_best_model_0001.pt',
    'load_model_path':None,
    'exp_name':'runs_summarization/representation_model',
    'debug':False,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}
