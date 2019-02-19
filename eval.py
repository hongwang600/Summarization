import torch
from utils import build_vocab, build_paragraph, filter_output, mask_sentence
from config import CONFIG as conf
from data_loader import get_train_dev_test_data, read_oracle, read_target_txt
import torch.nn as nn
from PyRouge.Rouge import Rouge

batch_size = conf['batch_size']
device = conf['device']
model_path = conf['model_path']
random_seed = conf['random_seed']
exp_name = conf['exp_name']
model_to_load = conf['load_model_path']
mask_pro = conf['mask_pro']
loss_margin = conf['loss_margin']
rouge_calculator = Rouge.Rouge(use_ngram_buf=True)
summarizer_model_path = conf['summarizer_model_path']
dev_oracle_file = conf['dev_oracle_file']
dev_tgt_text_file = conf['dev_tgt_text_file']
test_tgt_text_file = conf['test_tgt_text_file']

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
            target_idx = torch.arange(mask_size).to(device)
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

def evaluate_summarizer(model, data, labels, my_vocab, target_src):
    all_paragraphs = [build_paragraph(this_sample, my_vocab)
                      for this_sample in data]
    all_paragraph_lengths = [len(this_sample) for this_sample in data]
    sel_top_k = 3
    acc_total = 0
    recall_total = 0
    correct_total = 0
    predict_txt = []
    for current_batch in range(int((len(data)-1)/batch_size) + 1):
        batch_data = data[current_batch*batch_size:
                                (current_batch+1)*batch_size]
        paragraphs = all_paragraphs[current_batch*batch_size:
                                (current_batch+1)*batch_size]
        paragraph_lengths = all_paragraph_lengths[current_batch*batch_size:
                                (current_batch+1)*batch_size]
        scores = model(paragraphs)
        if labels is not None:
            targets = labels[current_batch*batch_size:
                                   (current_batch+1)*batch_size]
            _, pred_idx = scores.topk(sel_top_k, -1)
            for i, this_target in enumerate(targets):
                acc_total += len(this_target)
                recall_total += sel_top_k
                correct_total += len([pred for pred in pred_idx[i]
                                    if pred in this_target])
                pred_sentences = [batch_data[i][j] for j in pred_idx[i]
                                  if j < len(batch_data[i])]
                if len(pred_sentences) == 0:
                    pred_sentences = batch_data[i][:sel_top_k]
                #print(pred_sentences)
                joined_sentences = [' '.join(sentence) for sentence in
                                    pred_sentences]
                predict_txt.append(' '.join(joined_sentences))
        else:
            _, pred_idx = scores.topk(sel_top_k, -1)
            for i in range(len(batch_data)):
                pred_sentences = [batch_data[i][j] for j in pred_idx[i]
                                  if j < len(batch_data[i])]
                if len(pred_sentences) == 0:
                    pred_sentences = batch_data[i][:sel_top_k]
                #print(pred_sentences)
                joined_sentences = [' '.join(sentence) for sentence in
                                    pred_sentences]
                predict_txt.append(' '.join(joined_sentences))

    scores = rouge_calculator.compute_rouge(target_src, predict_txt)

    if labels is not None:
        return float(correct_total)/acc_total, float(correct_total)/recall_total,\
            scores['rouge-2']['f'][0]
    else:
        return -1, -1, scores['rouge-2']['f'][0]

if __name__ == '__main__':
    train_data, dev_data, test_data = get_train_dev_test_data()
    my_vocab = build_vocab(train_data, dev_data, test_data)
    #train_oracle = read_oracle(train_oracle_file)
    dev_oracle = read_oracle(dev_oracle_file)
    dev_target_txt = read_target_txt(dev_tgt_text_file)
    test_target_txt = read_target_txt(test_tgt_text_file)
    model = torch.load(summarizer_model_path)
    print(evaluate_summarizer(model, dev_data, dev_oracle, my_vocab, dev_target_txt))
    print(evaluate_summarizer(model, test_data, None, my_vocab, test_target_txt))
