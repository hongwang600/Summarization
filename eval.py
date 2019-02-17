import torch
from utils import build_vocab, build_paragraph, filter_output, mask_sentence
from config import CONFIG as conf
import torch.nn as nn

batch_size = conf['batch_size']
device = conf['device']
model_path = conf['model_path']
random_seed = conf['random_seed']
exp_name = conf['exp_name']
model_to_load = conf['load_model_path']
mask_pro = conf['mask_pro']
loss_margin = conf['loss_margin']

def compute_score(outs, pool_sent_embeds, masks):
    cos = nn.CosineSimilarity(dim=-1)
    #print(outs)
    #print(pool_sent_embeds)
    all_pos_scores = []
    all_neg_scores = []
    num_corrects = 0
    num_samples = 0
    for i, mask in enumerate(masks):
        mask_idx = torch.arange(len(mask))[mask].long()
        mask_size = len(mask_idx)
        if mask_size > 0:
            mask_pos_out = outs[i][mask_idx]
            mask_sent_embeds = pool_sent_embeds[i]
            mask_pos_out = mask_pos_out.unsqueeze(1)
            scores = cos(mask_pos_out, mask_sent_embeds)
            _, pred_idx = scores.max(-1)
            target_idx = torch.arange(mask_size).cuda()
            num_samples += mask_size
            num_corrects += torch.sum(pred_idx==target_idx)
    return num_corrects, num_samples

def evaluate(model, data, my_vocab):
    all_paragraphs = [build_paragraph(this_sample, my_vocab)
                      for this_sample in data]
    all_paragraph_lengths = [len(this_sample) for this_sample in data]
    total_corrects = 0
    total_samples = 0
    for current_batch in range(int((len(data)-1)/batch_size) + 1):
        paragraphs = all_paragraphs[current_batch*batch_size:
                                (current_batch+1)*batch_size]
        paragraph_lengths = all_paragraph_lengths[current_batch*batch_size:
                                (current_batch+1)*batch_size]
        masked_paragraphs, masks, cand_pool = mask_sentence(paragraphs)
        outs, pool_sent_embeds = model(masked_paragraphs, cand_pool)
        corrects, batch_samples = compute_score(outs, pool_sent_embeds, masks)
        total_corrects += corrects
        total_samples += batch_samples
    return float(total_corrects)/total_samples
