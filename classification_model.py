import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, representation_model, dim_size, num_cata):
        super(ClassificationModel, self).__init__()
        self.embed_model = representation_model
        self.score_layer = nn.Linear(dim_size, num_cata)

    def forward(self, paragraphs):
        embeds = self.embed_model(paragraphs)
        batch_size, doc_size, embed_dim = embeds.size()
        #embeds = embeds.view(-1, embed_dim)
        #cls_embed = embeds[:,0]
        cls_embed, _ = embeds.max(1)
        #scores = sigmoid(self.score_layer(embeds).view(batch_size, doc_size))
        scores = self.score_layer(cls_embed)
        return scores
