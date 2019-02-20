from data_loader import get_train_dev_test_data, read_oracle, read_target_txt,\
    read_target_20_news
from utils import build_vocab, build_paragraph, filter_output, mask_sentence, \
    load_vocab
from config import CONFIG as conf
from model import MyModel
from classification_model import ClassificationModel
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from eval import evaluate, evaluate_summarizer, evaluate_classifier
from tensorboardX import SummaryWriter
import random

batch_size = conf['batch_size']
num_epoch = conf['epoch']
device = conf['device']
learning_rate = conf['learning_rate']
#model_path = conf['model_path']
random_seed = conf['random_seed']
exp_name = conf['exp_name']
mask_pro = conf['mask_pro']
loss_margin = conf['loss_margin']
hidden_dim = conf['hidden_dim']
train_tgt_file = conf['train_tgt_file']
dev_tgt_file = conf['dev_tgt_file']
classifier_model_path = conf['classifier_model_path']
classifier_embed_model_path = conf['classifier_embed_model_path']
num_classes = conf['num_classes']

torch.manual_seed(random_seed)
random.seed(random_seed)

def build_training_pairs(outs, pool_sent_embeds, masks):
    cos = nn.CosineSimilarity(dim=-1)
    #print(outs)
    #print(pool_sent_embeds)
    all_pos_scores = []
    all_neg_scores = []
    for i, mask in enumerate(masks):
        mask_idx = torch.arange(0, len(mask))[mask].long()
        mask_size = len(mask_idx)
        if mask_size > 0:
            mask_pos_out = outs[i][mask_idx]
            mask_sent_embeds = pool_sent_embeds[i]
            mask_pos_out = mask_pos_out.unsqueeze(1)
            scores = cos(mask_pos_out, mask_sent_embeds)
            eye_idx = torch.eye(mask_size).byte()
            n_eye_idx = (torch.ones(mask_size, mask_size).byte()-eye_idx).byte()
            pos_score = scores[eye_idx].view(-1,1).expand(-1, mask_size-1)
            pos_score = pos_score.contiguous().view(-1)
            neg_score = scores[n_eye_idx].view(-1)
            all_pos_scores.append(pos_score)
            #for i in range(len(pos_score)):
            #    all_pos_scores.append(pos_score[i])
            all_neg_scores.append(neg_score)
    return torch.cat(all_pos_scores), torch.cat(all_neg_scores)


def train(train_data, dev_data, my_vocab, train_target, dev_target):
    #model = None
    embed_model = MyModel(my_vocab)
    #model = nn.DataParallel(model)
    embed_model = embed_model
    if classifier_embed_model_path  is not None:
        embed_model = torch.load(classifier_embed_model_path)
    #criteria = torch.nn.CrossEntropyLoss()
    model = ClassificationModel(embed_model, hidden_dim*2, num_classes)
    model = model.to(device)
    #criteria = torch.nn.MSELoss()
    criteria = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=learning_rate)
    best_acc = -1
    writer = SummaryWriter(exp_name)
    #print(len(train_data))
    all_paragraphs = [build_paragraph(this_sample, my_vocab)
                      for this_sample in train_data]
    all_paragraph_lengths = [len(this_sample) for this_sample in train_data]
    train_idx = list(range(len(train_data)))
    for epoch_i in range(num_epoch):
        random.shuffle(train_idx)
        total_loss = 0
        total_batch = 0
        all_paragraphs = [all_paragraphs[i] for i in train_idx]
        all_paragraph_lengths = [all_paragraph_lengths[i] for i in train_idx]
        train_target = [train_target[i] for i in train_idx]
        for current_batch in range(int((len(train_data)-1)/batch_size) + 1):
            if current_batch%100 ==0:
                print(current_batch)
            model_optim.zero_grad()
            paragraphs = all_paragraphs[current_batch*batch_size:
                                    (current_batch+1)*batch_size]
            paragraph_lengths = all_paragraph_lengths[current_batch*batch_size:
                                    (current_batch+1)*batch_size]
            scores = model(paragraphs)
            targets = train_target[current_batch*batch_size:
                                   (current_batch+1)*batch_size]
            labels = torch.tensor(targets).to(device)
            loss = criteria(scores, labels)
            #print(loss)
            total_loss += loss.item()
            total_batch += 1
            loss.backward()
            model_optim.step()
        acc = evaluate_classifier(model, dev_data, dev_target, my_vocab)
        if acc > best_acc:
            torch.save(model, classifier_model_path)
            best_acc = acc
        writer.add_scalar('accuracy', acc, epoch_i)

if __name__ == '__main__':
    train_data, dev_data, test_data = get_train_dev_test_data()
    #print(train_data[0])
    #print(dev_data[0])
    #print(test_data[0])
    #my_vocab = build_vocab([train_data, test_data])
    my_vocab = load_vocab()
    print('vocab loaded')
    train_target = read_target_20_news(train_tgt_file)
    dev_target = read_target_20_news(dev_tgt_file)
    train(train_data, dev_data, my_vocab, train_target, dev_target)
