import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        num_logits = logits_per_image.shape[0]
        labels = torch.eye(num_logits, device=device, dtype=torch.float)
        pred_1 = F.log_softmax(logits_per_image, dim=-1)
        pred_2 = F.log_softmax(logits_per_text, dim=-1)
        loss_a = F.cross_entropy(pred_1, labels, reduction='sum') / num_logits
        loss_b = F.cross_entropy(pred_2, labels, reduction='sum') / num_logits
        total_loss = (loss_a + loss_b) / 2
        return total_loss
