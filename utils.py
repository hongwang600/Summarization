import gluonnlp
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from config import CONFIG as conf
import random
import json

mask_pro = conf['mask_pro']
random_seed = conf['random_seed']
device = conf['device']
random.seed(random_seed)
torch.manual_seed(random_seed)

def get_all_words(data):
    all_text = []
    #print([data[0][1]])
    for sample in data:
        for sentence in sample:
            all_text += sentence
    return all_text

def build_vocab(data_list):
    all_data = []
    for dataset in data_list:
        all_data += dataset
    #all_data = train_data + dev_data + test_data
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

def save_vocab(vocab):
    json_str = vocab.to_json()
    with open('data/vocab.json', 'w') as file_out:
        json.dump(json_str, file_out)

def load_vocab():
    with open('data/vocab.json') as file_in:
        json_str = json.load(file_in)
    vocab = gluonnlp.Vocab.from_json(json_str)
    glove = gluonnlp.embedding.create('glove', source='glove.6B.100d')
    vocab.set_embedding(glove)
    return vocab

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
    return new_batch_data, batch_mask, cand_pool

def switch_within_para(mask, para_embed, para_len):
    para_size = len(para_len)
    cand_idx = [i for i in range(para_size) if mask[i]==1]
    origin_idx = list(cand_idx)
    random.shuffle(cand_idx)
    for i, idx in enumerate(cand_idx):
        if idx == origin_idx[i] and i!=0:
            cand_idx[i], cand_idx[i-1] = cand_idx[i-1], cand_idx[i]
        elif idx==origin_idx[i]:
            cand_idx[i], cand_idx[i+1] = cand_idx[i+1], cand_idx[i]
    new_cand_embed = list(para_embed)
    new_cand_len = list(para_len)
    for i, idx in enumerate(origin_idx):
        new_cand_embed[idx] = para_embed[cand_idx[i]]
        new_cand_len[idx] = para_len[cand_idx[i]]
    return new_cand_embed, new_cand_len

def local_sort_sentence(batch_data, cand_permutation):
    sorter_len = 3
    new_batch_data = []
    cand_labels = torch.LongTensor(len(batch_data)).random_(0, len(cand_permutation))
    sel_labels = []
    start_idx_list = []
    for i, para in enumerate(batch_data):
        if len(para[0]) < sorter_len:
            continue
        start_idx = random.randint(0, len(para[0])-sorter_len)
        start_idx_list.append(start_idx)
        para_embed = list(para[0])
        para_len = list(para[1])
        this_label = cand_labels[i]
        sel_labels.append(this_label)
        this_permut = cand_permutation[this_label]
        for j in range(len(this_permut)):
            para_embed[start_idx+j] = para[0][start_idx+this_permut[j]]
            para_len[start_idx+j] = para[1][start_idx+this_permut[j]]
        new_batch_data.append([para_embed, para_len])
    return new_batch_data, start_idx_list, sel_labels

def switch_sentence(batch_data, sentence_cands):
    batch_mask = []
    new_batch_data = []
    cand_pool = []
    cand_size = len(sentence_cands)
    #print(len(batch_data))
    for para in batch_data:
        if len(para[0]) < 2:
            #batch_mask.append(torch.zeros(len(para[0])).byte())
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
        para_embed, para_len = switch_within_para(mask, para_embed, para_len)
        new_batch_data.append([para_embed, para_len])
    return new_batch_data, batch_mask

def replace_sentence(batch_data, sentence_cands):
    batch_mask = []
    new_batch_data = []
    cand_pool = []
    cand_size = len(sentence_cands)
    #print(len(batch_data))
    for para in batch_data:
        if len(para[0]) < 2:
            #batch_mask.append(torch.zeros(len(para[0])).byte())
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
                #this_cand_pool_embed.append(para_embed[i])
                #this_cand_pool_length.append(para_len[i])
                para_embed[i] = sentence_cands[random.randint(0, cand_size-1)]
                para_len[i] = len(para_embed[i])
        new_batch_data.append([para_embed, para_len])
    return new_batch_data, batch_mask

def gen_mask_based_length(batch_size, doc_size, lengths):
    masks = torch.ones(batch_size, doc_size)
    index_matrix = torch.arange(0, doc_size).expand(batch_size, -1)
    index_matrix = index_matrix.long()
    doc_lengths = torch.tensor(lengths).view(-1,1)
    doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
    masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
    return masks.to(device)

def get_fetch_idx(batch_size, start_idx):
    num_to_sort = 3
    start_idx = torch.tensor(start_idx)
    idx_1 = torch.arange(batch_size).view(-1,1).expand(-1, num_to_sort)
    idx_1 = idx_1.contiguous().view(-1)
    idx_2 = start_idx.view(-1,1).expand(-1, num_to_sort)
    idx_2 = idx_2.contiguous()
    for i in range(num_to_sort):
        idx_2[:,i] += i
    idx_2 = idx_2.contiguous().view(-1)
    return idx_1, idx_2
