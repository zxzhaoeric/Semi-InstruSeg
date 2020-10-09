import torch
from torch import nn
import numpy as np
import tqdm

from metrics import (
    iou_binary_torch,
    iou_binary_3d,
    dice_binary_torch,
    iou_multi_np,
    dice_multi_np,
    )


class LossMulti:
    """
    Loss = (1 - jaccard_weight) * CELoss - jaccard_weight * log(JaccardLoss)
    """

    def __init__(self, num_classes, jaccard_weight, class_weights=None):
        if class_weights is not None:
            weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            weight = None

        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        '''
        @param outputs: n-d tensor (probability map)
        @param targets: tensor (0-1 map)

        @return loss: loss
        '''
        loss = (1 - self.jaccard_weight) * self.ce_loss(outputs, targets)

        jaccard_loss = 0
        # turn logits into prob maps
        outputs = torch.softmax(outputs, dim=1)

        if self.jaccard_weight > 0:
            # for each class, calculate jaccard loss
            for cls in range(self.num_classes):
                metric = iou_binary_torch((targets == cls).float(), outputs[:, cls])
                # metric = iou_binary_dice((targets == cls).float(), outputs[:, cls:cls+1])

                # mean of metric(class = cls) for all inputs
                jaccard_loss += torch.mean(metric)
            loss -= self.jaccard_weight * torch.log(jaccard_loss / self.num_classes)
        return loss


class Loss3d:
    """
    Loss = (1 - jaccard_weight) * CELoss - jaccard_weight * log(JaccardLoss)
    """

    def __init__(self, num_classes, jaccard_weight, class_weights=None):
        if class_weights is not None:
            weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            weight = None

        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        '''
        @param outputs: n-d tensor (probability map)
        @param targets: tensor (0-1 map)

        @return loss: loss
        '''
        loss = (1 - self.jaccard_weight) * self.ce_loss(outputs, targets)

        jaccard_loss = 0
        # turn logits into prob maps
        outputs = torch.softmax(outputs, dim=1)

        if self.jaccard_weight > 0:
            # for each class, calculate jaccard loss
            for cls in range(self.num_classes):
                metric = iou_binary_3d((targets == cls).float(), outputs[:, cls])
                # metric = iou_binary_dice((targets == cls).float(), outputs[:, cls:cls+1])

                # mean of metric(class = cls) for all inputs
                jaccard_loss += torch.mean(metric)
            loss -= self.jaccard_weight * torch.log(jaccard_loss / self.num_classes)
        return loss
