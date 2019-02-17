import torch
CONFIG= {
    'learning_rate': 0.0001,
    'embedding_dim': 100,
    'hidden_dim': 200,
    'batch_size': 64,
    'eval_batch_size': 100,
    'epoch': 300,
    'random_seed': 1,
    'train_file':'data/arxiv/train.json',
    'dev_file':'data/arxiv/dev.json',
    'test_file':'data/arxiv/test.json',
    'model_path':'results/arxiv_pointerNet_true_label.pt',
    'load_model_path':None,
    'exp_name':'runs_arxiv/arxiv_pointerNet_true_label',
    'debug':False,
    'device': torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
}
