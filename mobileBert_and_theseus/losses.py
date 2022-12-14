from __future__ import print_function

import torch
import torch.nn as nn


# Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
# It also supports the unsupervised contrastive loss in SimCLR
class SupConLoss(nn.Module):
    def __init__(self, temperature=10, contrast_mode='all', base_temperature=100):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # get mask through labels
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, torch.transpose(labels, 1, 0)).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_feature = anchor_feature.unsqueeze(1)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(anchor_feature, 1, 0)), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        anchor_dot_contrast_den = anchor_dot_contrast*1.05
        logits_max_den, _ = torch.max(anchor_dot_contrast_den, dim=1, keepdim=True)
        logits_den = anchor_dot_contrast_den - logits_max_den.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits_den) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # smoothing
        mean_log_prob_pos = ((mask * log_prob).sum(1)) / (mask.sum(1)+1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
