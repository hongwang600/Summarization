import json
from config import CONFIG as conf
from ast import literal_eval as make_tuple

#train_file = 'data/train.json'
#dev_file = 'data/dev.json'
#test_file = 'data/test.json'
train_file = conf['train_file']
dev_file = conf['dev_file']
test_file = conf['test_file']
max_sent_len = 100
max_doc_len = 80

def read_data(filename, add_first_sentence, keep_single_sent):
    data = []
    with open(filename) as in_file:
        for line in in_file:
            line = line.strip()
            all_sentences = []
            if add_first_sentence:
                all_sentences = [['<startsent>']]
            count = len(all_sentences)
            for sentence in line.split('##SENT##'):
                sentence = sentence.split()[:max_sent_len]
                all_sentences.append(sentence)
                count+=1
                if count == max_doc_len:
                    break
            if keep_single_sent or len(all_sentences) > 1:
                data.append(all_sentences)
    return data

#if __name__ == '__main__':
def get_train_dev_test_data(add_first_sentence = False, keep_single_sent=True):
    train_data = read_data(train_file, add_first_sentence, keep_single_sent)
    dev_data = read_data(dev_file, add_first_sentence, keep_single_sent)
    test_data = read_data(test_file, add_first_sentence, keep_single_sent)
    return train_data, dev_data, test_data
    #print(len(train_data))
    #print(train_data[0])

def read_oracle(file_name):
    target = []
    with open(file_name) as in_file:
        for line in in_file:
            oracle_tuple = make_tuple(line.split('\t')[0])
            if oracle_tuple is not None:
                oracle_tuple = list(oracle_tuple)
                new_oracle_tuple = [i for i in oracle_tuple if i < max_doc_len]
                oracle_tuple = new_oracle_tuple
            else:
                oracle_tuple = []
            target.append(oracle_tuple)
    return target

def read_target_txt(file_name):
    target = []
    with open(file_name) as in_file:
        for line in in_file:
            line = line.strip()
            sentences = line.split('##SENT##')
            target.append(' '.join(sentences))
    return target

def read_target_20_news(file_name):
    target = []
    with open(file_name) as in_file:
        for line in in_file:
            line = line.strip()
            target.append(int(line))
    return target
