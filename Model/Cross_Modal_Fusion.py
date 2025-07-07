from torch import nn, Tensor

class CrossModalFusion(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossModalFusion, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, text_features: Tensor, visual_features: Tensor):
        # Self attention on the text features
        text_features = text_features.permute(1, 0, 2)  # Convert to (seq_len, batch_size, embed_dim) for MultiheadAttention
        text_self_out, self_attn_weights = self.self_attention(text_features, text_features, text_features)
        text_self_out = text_features + text_self_out  # Add & Norm step
        text_self_out = self.norm1(text_self_out.permute(1, 0, 2))  # Convert back to (batch_size, seq_len, embed_dim)

        # Cross attention with the output of self-attention and visual features
        visual_features = visual_features.permute(1, 0, 2)  # Convert to (seq_len, batch_size, embed_dim)
        text_cross_out, cross_attn_weights  = self.cross_attention(text_self_out.permute(1, 0, 2), visual_features, visual_features)
        text_cross_out = text_self_out + text_cross_out.permute(1, 0, 2)  # Add & Norm step
        text_cross_out = self.norm2(text_cross_out)

        # Feed Forward Network
        text_ff_out = self.feed_forward(text_cross_out)
        text_ff_out = text_cross_out + text_ff_out  # Add & Norm step
        text_ff_out = self.norm3(text_ff_out)

        return text_ff_out

class MultiLayerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(MultiLayerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CrossModalFusion(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def attn_forward(self, src, src2):
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            src, self_attn_weights, cross_attn_weights = layer(src, src2)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
        src = self.norm(src)
        return src, self_attn_weights_list, cross_attn_weights_list

    def forward(self, src, src2):
        for layer in self.layers:
            src = layer(src, src2)
        src = self.norm(src)
        return src

