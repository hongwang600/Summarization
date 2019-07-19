import torch
basic_path = '/mnt/sshd/hongwang/results/'
#basic_name = 'basic'
#basic_name = 'mask'
#basic_name = 'replace'
#basic_name = 'switch'
#basic_name = 'local_sort'
#basic_name = 'switch_one_epoch'
#basic_name = 'newsroom_switch'
basic_name = 'switch_0_35'
#embed_path = 'results/useful_represents/mask_0_25_representation_model.pt'
#embed_path = 'results/useful_represents/replace_0_25_representation_model.pt'
#embed_path = 'results/useful_represents/switch_0_25_representation_model.pt'
#embed_path = 'results/useful_represents/switch_one_epoch_representation_model.pt'
#embed_path = 'results/useful_represents/local_sort_representation_model.pt'
#embed_path = 'results/useful_represents/aug_data_switch_representation_model.pt'
#embed_path = 'results/useful_represents/newsroom_switch_representation_model.pt'
embed_path = 'results/pretrain_model.pt'
CONFIG= {
    'learning_rate': 0.00001,
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
    'model_path':'results/representation_model.pt',
    'summarizer_model_path':basic_path + 'models/'+basic_name+'_model',
    'score_path':basic_path + 'scores/'+basic_name+'_model',
    'summarizer_embed_model_path': embed_path,
    #'summarizer_embed_model_path':None,
    'ref_folder': basic_path+'eval/ref_'+basic_name+'/',
    'pred_folder': basic_path+'eval/pred_'+basic_name+'/',
    #'load_model_path':'results/sind_best_model_0001.pt',
    'load_model_path':None,
    'exp_name':basic_path+'reruns_summarization/'+basic_name,
    'debug':False,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}
