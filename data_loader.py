import json
from config import CONFIG as conf

#train_file = 'data/train.json'
#dev_file = 'data/dev.json'
#test_file = 'data/test.json'
train_file = conf['train_file']
dev_file = conf['dev_file']
test_file = conf['test_file']

def read_data(filename):
    data = []
    with open(filename) as in_file:
        for line in in_file:
            line = line.strip()
            all_sentences = []
            for sentence in line.split('##SENT##'):
                sentence = sentence.split()
                all_sentences.append(sentence)
            data.append(all_sentences)
    return data

#if __name__ == '__main__':
def get_train_dev_test_data():
    train_data = read_data(train_file)
    dev_data = read_data(dev_file)
    test_data = read_data(test_file)
    return train_data, dev_data, test_data
    #print(len(train_data))
    #print(train_data[0])
