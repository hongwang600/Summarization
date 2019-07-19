import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from config import CONFIG as conf
import numpy as np
import torch.nn.functional as F
from attention import SimpleEncoder

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
device = conf['device']
random_seed = conf['random_seed']

torch.manual_seed(random_seed)


class Packed(nn.Module):
    '''
    usage:
    initialize your LSTM as lstm = Packed(nn.LSTM(...))
    '''

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    @property
    def batch_first(self):
        return self.rnn.batch_first

    def forward(self, inputs, lengths, hidden=None, max_length=None):
        lengths = torch.tensor(lengths)
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices]
        outputs, (h, c) = self.rnn(nn.utils.rnn.pack_padded_sequence(inputs, lens.tolist(), batch_first=self.batch_first), hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first, total_length=max_length)
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(vocab_embedding))
        self.lstm = Packed(nn.LSTM(embedding_dim, hidden_dim,
                                   bidirectional=True))

    def forward(self, padded_sentences, lengths):
        padded_embeds = self.embedding(padded_sentences)
        #print(len(padded_sentences))
        lstm_out, hidden_state = self.lstm(padded_embeds, lengths)
        permuted_hidden = hidden_state[0].permute([1,0,2]).contiguous()
        return permuted_hidden.view(-1, self.hidden_dim*2)

class MyModel(nn.Module):
    def __init__(self, my_vocab):
        super(MyModel, self).__init__()
        my_embed = my_vocab.embedding.idx_to_vec
        self.sentence_encoder = BiLSTM(len(my_embed), my_embed.asnumpy())
        self.self_attention = SimpleEncoder(hidden_dim*2, 4, 5)

    def pack_paragraph(self, paragraphs):
        paragraph_lengths = []
        sentence_lengths = []
        sentences = []
        for para in paragraphs:
            paragraph_lengths.append(len(para[1]))
            #print(para[0])
            sentences += para[0]
            sentence_lengths += para[1]
        return paragraph_lengths, sentence_lengths, sentences

    def unpack_paragraph(self, embeds, lengths):
        ret_embeds = []
        start_index = 0
        for i in range(len(lengths)):
            ret_embeds.append(embeds[start_index:start_index+lengths[i]])
            start_index += lengths[i]
        return ret_embeds

    def mask_lengths(self, batch_size, doc_size, lengths):
        masks = torch.ones(batch_size, doc_size)
        index_matrix = torch.arange(0, doc_size).expand(batch_size, -1)
        index_matrix = index_matrix.long()
        doc_lengths = torch.tensor(lengths).view(-1,1)
        doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
        masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
        return masks.to(device)

    def encode_sentences(self, paragraphs):
        paragraph_lengths, sentence_lengths, sentences = \
            self.pack_paragraph(paragraphs)
        batch_size = len(paragraph_lengths)
        doc_size = max(paragraph_lengths)
        #print(sentences)
        padded_sentences = pad_sequence(sentences, padding_value=1).long().to(device)
        sentence_embeds = self.sentence_encoder(padded_sentences,
                                                sentence_lengths)
        paragraph_embeds = self.unpack_paragraph(sentence_embeds,
                                                 paragraph_lengths)
        return paragraph_embeds, paragraph_lengths

    def forward(self, paragraphs, cand_pool=None):
        #print(paragraph_embeds)
        batch_size = len(paragraphs)
        if cand_pool is not None:
            paragraph_embeds, paragraph_lengths = \
                self.encode_sentences(paragraphs+cand_pool)
            cand_pool_embeds = paragraph_embeds[batch_size:]
            paragraph_embeds = paragraph_embeds[:batch_size]
            paragraph_lengths = paragraph_lengths[:batch_size]
        else:
            paragraph_embeds, paragraph_lengths = \
                self.encode_sentences(paragraphs)
        doc_size = max(paragraph_lengths)
        padded_paragraph_embeds = pad_sequence(paragraph_embeds,
                                               batch_first=True)
        masks = self.mask_lengths(batch_size, doc_size, paragraph_lengths)
        outs = self.self_attention(padded_paragraph_embeds, masks)
        if cand_pool is not None:
            return outs, cand_pool_embeds
        else:
            return outs
        #return paragraph_embeds, cand_pool_embeds
