import os
import numpy as np
from typing import Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, BertConfig,AutoTokenizer

torch.autograd.set_detect_anomaly(True)

class text_encoder(nn.Module):
    def __init__(self, bert_model_name: str, embed_dim: int = 768, freeze_layers: Union[Tuple[int, int], int] = None):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):
        try:
            if os.path.exists(bert_model_name):
                config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
                model = AutoModel.from_pretrained(bert_model_name, config=config)
            else:
                config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
                model = AutoModel.from_pretrained(bert_model_name, config=config)
            print("Text feature extractor:", bert_model_name)
            print("BERT encoder layers:", len(model.encoder.layer))
        except Exception as e:
            raise Exception(f"Failed to load model {bert_model_name} with error: {str(e)}")

        if freeze_layers is not None:
            if isinstance(freeze_layers, int):
                freeze_layers = [freeze_layers]
            for layer_idx in freeze_layers:
                for param in model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        output = self.bert_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        last_hidden_state, pooler_output, hidden_states = output[0], output[1], output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out, last_hidden_state

    def forward(self, text):

        text_features, text_features_mask = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return text_features, text_features_mask, self.logit_scale.exp()




