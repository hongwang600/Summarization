import torch
CONFIG= {
    'learning_rate': 0.0001,
    'embedding_dim': 100,
    'hidden_dim': 200,
    'batch_size': 20,
    'eval_batch_size': 100,
    'epoch': 300,
    'random_seed': 1,
    'train_file':'data/sind/train.json',
    'dev_file':'data/sind/dev.json',
    'test_file':'data/sind/test.json',
    'out_train_file':'data/sind/train_1.json',
    'out_dev_file':'data/sind/dev_1.json',
    'out_test_file':'data/sind/test_1.json',
    'model_path':'results/sind_pointerNet_reward.pt',
    #'load_model_path':'results/sind_best_model_0001.pt',
    'load_model_path':None,
    'exp_name':'runs_sind/sind_pointerNet_reward',
    'debug':False,
    'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
}
