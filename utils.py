import gluonnlp
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from config import CONFIG as conf
import random

mask_pro = conf['mask_pro']
random_seed = conf['random_seed']
random.seed(random_seed)
torch.manual_seed(random_seed)

def get_all_words(data):
    all_text = []
    #print([data[0][1]])
    for sample in data:
        for sentence in sample:
            all_text += sentence
    return all_text

def build_vocab(train_data, dev_data, test_data):
    all_data = train_data + dev_data + test_data
    all_text = get_all_words(all_data)
    #print(all_text[:5])
    counter = gluonnlp.data.count_tokens(all_text)
    my_vocab = gluonnlp.Vocab(counter)
    glove = gluonnlp.embedding.create('glove', source='glove.6B.100d')
    my_vocab.set_embedding(glove)
    #my_embed = my_vocab.embedding.idx_to_vec
    #print(my_vocab.embedding[['hello', 'world']][:, :5])
    #index = my_vocab[['hello', 'world']]
    #print(index)
    #print(my_embed[index][:, :5])
    #print(len(my_embed))
    #print(my_embed[1])
    #print(my_vocab.idx_to_token[1])
    #print(my_vocab.idx_to_token[0])
    #print(my_vocab.idx_to_token[2])
    #embed = nn.Embedding(len(my_embed), len(my_embed[0]))
    #embed.weight.data.copy_(torch.from_numpy(my_embed.asnumpy()))
    return my_vocab

def build_paragraph(text_data, my_vocab):
    #print(my_vocab[text_data[0]])
    #print(text_data)
    indexs = [torch.tensor(my_vocab[text]).long() for text in text_data]
    lengths = [len(text) for text in text_data]
    #padded_sequence = pad_sequence(indexs, padding_value=1)
    return indexs, lengths

def filter_output(x, lengths):
    batch_size = max(lengths)
    indexes = []
    for i in range(len(lengths)):
        indexes += [i*batch_size+j for j in range(lengths[i])]
    #print(len(indexes), sum(lengths))
    #print(lengths, indexes)
    return x[indexes]

def mask_sentence(batch_data):
    mask_id = torch.zeros(1)
    batch_mask = []
    new_batch_data = []
    cand_pool = []
    #print(len(batch_data))
    for para in batch_data:
        if len(para[0]) < 2:
            continue
        para_embed = list(para[0])
        para_len = list(para[1])
        this_cand_pool_embed = []
        this_cand_pool_length = []
        mask = torch.rand(len(para_len))
        mask = mask.le(mask_pro)
        if mask.sum() <=1:
            idx = list(range(len(mask)))
            sel_idx = random.sample(idx, 2)
            mask[sel_idx] = 1
            #mask[random.randint(0,len(mask)-1)]=1
        batch_mask.append(mask)
        for i in range(len(mask)):
            if mask[i] == 1:
                this_cand_pool_embed.append(para_embed[i])
                this_cand_pool_length.append(para_len[i])
                para_embed[i] = mask_id
                para_len[i] = 1
        new_batch_data.append([para_embed, para_len])
        cand_pool.append([this_cand_pool_embed, this_cand_pool_length])
    '''
    for mask in batch_mask:
        mask.requires_grad=False
    for data in new_batch_data+cand_pool:
        for embed in data[0]:
            embed.requires_grad=False
    '''
    return new_batch_data, batch_mask, cand_pool
