import torch
import torch.nn as nn


def ITM_task(images_batch, reports_batch):
    batch_size = len(images_batch)
    labels_batch = torch.ones(batch_size, batch_size, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                labels_batch[i, j] = 0
    return  labels_batch


def ITMLoss(vision_features, text_features, labels):
        criterion = nn.BCEWithLogitsLoss()
        similarity = torch.matmul(vision_features, text_features.T)
        logits = similarity.view(-1)
        labels = labels.view(-1)
        loss = criterion(logits, labels)
        return loss

