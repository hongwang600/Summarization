import torch.nn as nn
import torch

class LocalSorterModel(nn.Module):
    def __init__(self, dim_size, num_items):
        super(LocalSorterModel, self).__init__()
        self.pairwise_layer = nn.Linear(2*dim_size, dim_size)
        num_pair = num_items*(num_items-1)
        num_results = 1
        for i in range(num_items):
            num_results *= (i+1)
        self.classify_layer = nn.Linear(int(num_pair*dim_size), int(num_results))

    def build_pairs(self, embeds):
        batch_size, num_items, embed_dim = embeds.size()
        rep_1 = embeds.unsqueeze(-2).expand(batch_size, num_items, num_items,
                                            embed_dim)
        rep_2 = embeds.unsqueeze(1).expand(batch_size, num_items, num_items,
                                           embed_dim)
        rep_1 = rep_1.contiguous().view(batch_size, -1, embed_dim)
        rep_2 = rep_2.contiguous().view(batch_size, -1, embed_dim)
        combined_embed = torch.cat((rep_1, rep_2), -1).view(batch_size,
                                                             -1, embed_dim*2)
        sel_mask = torch.ones(num_items, num_items)-torch.eye(num_items)
        sel_mask = sel_mask.view(-1).byte()
        #print(sel_mask)
        #print(combined_embed.size())
        #print(combined_embed)
        combined_embed = combined_embed[:,sel_mask, :]
        pairwise_embeds = self.pairwise_layer(combined_embed.view(-1,
                                                                  embed_dim*2))
        return pairwise_embeds


    def forward(self, embeds):
        batch_size, num_items, embed_dim = embeds.size()
        pairwise_embeds = self.build_pairs(embeds)
        pairwise_embeds = pairwise_embeds.view(batch_size, -1)
        return self.classify_layer(pairwise_embeds)
