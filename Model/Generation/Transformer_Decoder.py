import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask)[0])

        tgt2 = self.norm2(tgt)
        tgt,_=self.multihead_attn(tgt2.permute(1, 0, 2), memory.permute(1, 0, 2), memory.permute(1, 0, 2), attn_mask=memory_mask,
                                                      key_padding_mask=memory_key_padding_mask)
        tgt=tgt.permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt)


        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(tgt2)))))
        return tgt


class HKRGDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, vocab_size=10000, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = tgt
        for i, layer in enumerate(self.layers):
            memory = memory_list[i // 4]
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        output = self.norm(output)
        output = self.fc_out(output)

        return output


