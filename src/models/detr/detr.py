import os
from typing import Dict, Union, List, Tuple
import torch
from torch import nn, Tensor
from torch.quantization import quantize_dynamic

from src.models.detr.backbone import tiny, small, base, buildBackbone
from src.utils.misc import PostProcess
from src.models.mlp import MLP

class DETR(nn.Module):
    def __init__(self, args):
        super(DETR, self).__init__()

        # Select the appropriate Transformer
        if args.detr.transformerName == 'tiny':
            self.transformer, hidden_dim = tiny()
        elif args.detr.transformerName == 'small':
            self.transformer, hidden_dim = small()
        elif args.detr.transformerName == 'base':
            self.transformer, hidden_dim = base()
        else:
            raise ValueError(f"Backbone '{args.detr.transformerName}' not supported")

        self.backbone = buildBackbone(args, hidden_dim)
        self.reshape = nn.Conv2d(self.backbone.backbone.outChannels, hidden_dim, 1)

        # Keep these configurable through args
        self.queryEmbed = nn.Embedding(args.detr.numQuery, hidden_dim)
        self.class_embed = MLP(hidden_dim, hidden_dim, args.numClass + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, x: Tensor, meta=None) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        """
        :param x: tensor of shape [batchSize, 3, imageHeight, imageWidth].

        :return: a dictionary with the following elements:
            - class: the classification results for all queries with shape [batchSize, numQuery, numClass + 1].
                     +1 stands for no object class.
            - bbox: the normalized bounding box for all queries with shape [batchSize, numQuery, 4],
                    represented as [centerX, centerY, width, height].

        mask: provides specified elements in the key to be ignored by the attention.
              the positions with the value of True will be ignored
              while the position with the value of False will be unchanged.
              Since I am only training with images of the same shape, the mask should be all False.
              Modify the mask generation method if you would like to enable training with arbitrary shape.
        """
        features, (pos, mask) = self.backbone(x)
        features = self.reshape(features)

        N = features.shape[0]
        features = features.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)
        query = self.queryEmbed.weight
        query = query.unsqueeze(1).repeat(1, N, 1)

        out = self.transformer(features, mask, query, pos)

        outputsClass = self.class_embed(out)
        outputsCoord = self.bbox_embed(out).sigmoid()

        return {'class': outputsClass[-1],
                'bbox': outputsCoord[-1],
                'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputsClass[:-1], outputsCoord[:-1])]}


@torch.no_grad()
def buildInferenceModel(args, quantize=False):
    assert os.path.exists(args.weight), 'inference model should have pre-trained weight'
    device = torch.device(args.device)

    model = DETR(args).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))

    postProcess = PostProcess().to(device)

    wrapper = DETRWrapper(model, postProcess).to(device)
    wrapper.eval()

    if quantize:
        wrapper = quantize_dynamic(wrapper, {nn.Linear})

    print('optimizing model for inference...')
    return torch.jit.trace(wrapper, (torch.rand(1, 3, args.targetHeight, args.targetWidth).to(device),
                                     torch.as_tensor([args.targetWidth, args.targetHeight]).unsqueeze(0).to(device)))

class DETRWrapper(nn.Module):
    """ A simple DETR wrapper that allows torch.jit to trace the module since dictionary output is not supported yet """

    def __init__(self, detr, postProcess):
        super(DETRWrapper, self).__init__()

        self.detr = detr
        self.postProcess = postProcess

    def forward(self, x: Tensor, imgSize: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: batch images of shape [batchSize, 3, args.targetHeight, args.targetWidth] where batchSize equals to 1
        If tensor with batchSize larger than 1 is passed in, only the first image prediction will be returned

        :param imgSize: tensor of shape [batchSize, imgWidth, imgHeight]

        :return: the first image prediction in the following order: scores, labels, boxes.
        """

        out = self.detr(x)
        out = self.postProcess(out, imgSize)[0]
        return out['scores'], out['labels'], out['boxes']