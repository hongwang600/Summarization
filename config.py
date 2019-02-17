import torch
CONFIG= {
    'learning_rate': 0.0001,
    'embedding_dim': 100,
    'hidden_dim': 200,
    'batch_size': 20,
    'eval_batch_size': 100,
    'epoch': 300,
    'random_seed': 1,
    'train_file':'data/recipe/train.json',
    'dev_file':'data/recipe/dev.json',
    'test_file':'data/recipe/test.json',
    'out_train_file':'data/recipe/train_1.json',
    'out_dev_file':'data/recipe/dev_1.json',
    'out_test_file':'data/recipe/test_1.json',
    'model_path':'results/recipe_debug.pt',
    #'load_model_path':'results/recipe_best_model_0001.pt',
    'load_model_path':None,
    'exp_name':'runs_recipe/debug',
    'debug':False,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}
