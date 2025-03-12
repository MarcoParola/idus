from src.utils import boxOps

from torch import Tensor
from typing import Dict, Union, List

from src.models.yolos.backbone import *
from src.models.mlp import MLP


class Yolos(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.yolos.backboneName == 'tiny':
            self.backbone, hidden_dim = tiny(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        elif args.yolos.backboneName == 'small':
            self.backbone, hidden_dim = small(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        elif args.yolos.backboneName == 'base':
            self.backbone, hidden_dim = base(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        elif args.yolos.backboneName == 'small_dWr':
            self.backbone, hidden_dim = small_dWr(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        else:
            raise ValueError(f'backbone {args.yolos.backboneName} not supported')

        self.backbone.finetune_det(
            det_token_num=args.yolos.detTokenNum,
            img_size=args.yolos.init_pe_size,
            mid_pe_size=args.yolos.mid_pe_size,
            use_checkpoint=args.yolos.use_checkpoint)

        self.class_embed = MLP(hidden_dim, hidden_dim, args.numClass + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x: Tensor, meta=None) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:

        # Process the input through the backbone
        x = self.backbone(x)
        x = x.unsqueeze(0)

        # Create temporary variables for class and bbox outputs
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()

        # Temporary results in case you don't want to return them
        results = {
            'class': outputs_class[-1],
            'bbox': outputs_coord[-1],
            'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputs_class[:-1], outputs_coord[:-1])]
        }

        # Instead of returning results directly, you can handle them here or return a minimal output
        return results  # Return this for model testing or just modify based on use case


# Updated PostProcess (same, you can keep as is for post-processing if needed)
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the COCO API """

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of shape [batch_size x 2], containing the size of each image in the batch
                          (height, width) for scaling back bounding boxes.
        """
        # Accessing class probabilities and bounding boxes
        out_logits, out_bbox = outputs['class'], outputs['bbox']

        # Softmax over the class probabilities
        prob = torch.softmax(out_logits, -1)

        results = []
        for i in range(out_logits.shape[0]):  # Process each image independently
            # Extract scores and labels for the current image
            scores, labels = prob[i, :, :-1].max(-1)  # Exclude the background class

            # Convert bounding boxes to [x0, y0, x1, y1]
            boxes = boxOps.boxCxcywh2Xyxy(out_bbox[i])  # [100, 4]

            # Scale boxes to the original image size
            img_h, img_w = target_sizes[i]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            boxes = boxes * scale_fct

            # Store results for this image
            results.append({'scores': scores, 'labels': labels, 'boxes': boxes})

        return results
