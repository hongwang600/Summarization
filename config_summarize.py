import torch
CONFIG= {
    'learning_rate': 0.001,
    'embedding_dim': 100,
    'hidden_dim': 200,
    'batch_size': 64,
    'eval_batch_size': 64,
    'epoch': 300,
    'random_seed': 1,
    'mask_pro': 0.15,
    'loss_margin': 0.5,
    'train_file':'data/summarization/train.txt.src.10k',
    'dev_file':'data/summarization/val.txt.src.10k',
    'test_file':'data/summarization/test.txt.src',
    'out_train_file':'data/summarization/train_1.json',
    'out_dev_file':'data/summarization/dev_1.json',
    'out_test_file':'data/summarization/test_1.json',
    'model_path':'results/summarization_pointerNet_reward.pt',
    #'load_model_path':'results/sind_best_model_0001.pt',
    'load_model_path':None,
    'exp_name':'runs_summarization/summarization_pointerNet_reward',
    'debug':False,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}
