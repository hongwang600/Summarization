import torch.nn as nn

class SummarizeModel(nn.Module):
    def __init__(self, representation_model, dim_size):
        super(SummarizeModel, self).__init__()
        self.embed_model = representation_model
        self.score_layer = nn.Linear(dim_size, 1)

    def forward(self, paragraphs):
        embeds = self.embed_model(paragraphs)
        batch_size, doc_size, embed_dim = embeds.size()
        embeds = embeds.view(-1, embed_dim)
        sigmoid = nn.Sigmoid()
        scores = sigmoid(self.score_layer(embeds).view(batch_size, doc_size))
        return scores
