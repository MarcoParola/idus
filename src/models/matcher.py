from typing import Dict, List, Tuple
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor

from src.utils.box_ops import gIoU, boxCxcywh2Xyxy


class HungarianMatcher(nn.Module):
    def __init__(self, classCost: float = 1, bboxCost: float = 5, giouCost: float = 2):
        super(HungarianMatcher, self).__init__()
        self.classCost = classCost
        self.bboxCost = bboxCost
        self.giouCost = giouCost

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        batchSize, numQuery = x['class'].shape[:2]

        # Flatten outputs for all queries in the batch
        outProb = x['class'].flatten(0, 1).softmax(-1)  # [batchSize * numQuery, numClasses]
        outBbox = x['bbox'].flatten(0, 1)  # [batchSize * numQuery, 4]

        # Concatenate targets from all samples in the batch
        tgtIds = torch.cat([t['labels'] for t in y])  # [totalTarget]
        tgtBbox = torch.cat([t['boxes'] for t in y])  # [totalTarget, 4]

        # Debugging info
        numClasses = outProb.shape[-1]
        assert tgtIds.min() >= 0 and tgtIds.max() < numClasses, (
            f"Target IDs are out of bounds. Min: {tgtIds.min()}, Max: {tgtIds.max()}, NumClasses: {numClasses}"
        )

        # Compute individual cost components
        classLoss = -outProb[:, tgtIds]  # [batchSize * numQuery, totalTarget]
        bboxLoss = torch.cdist(outBbox, tgtBbox, p=1)  # [batchSize * numQuery, totalTarget]
        giouLoss = -gIoU(boxCxcywh2Xyxy(outBbox), boxCxcywh2Xyxy(tgtBbox))  # [batchSize * numQuery, totalTarget]

        # Combine cost components
        costMatrix = self.bboxCost * bboxLoss + self.classCost * classLoss + self.giouCost * giouLoss
        costMatrix = costMatrix.view(batchSize, numQuery, -1).cpu().detach()  # [batchSize, numQuery, totalTarget]

        # Ensure target sizes are valid for splitting
        sizes = [len(t['boxes']) for t in y]
        assert sum(sizes) == costMatrix.shape[-1], (
            f"Mismatch between total target size and cost matrix last dimension. Sizes: {sizes}, "
            f"CostMatrix dim: {costMatrix.shape[-1]}"
        )

        # Solve the assignment problem for each batch element
        ids = [
            linear_sum_assignment(c[i])  # Hungarian algorithm for each batch
            for i, c in enumerate(costMatrix.split(sizes, -1))
        ]

        # Convert to tensors
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in ids]
