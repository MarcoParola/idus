import os

import hydra
from omegaconf import DictConfig

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
        x = self.backbone(x)
        x = x.unsqueeze(0)
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        
        return {'class': outputs_class[-1],
                'bbox': outputs_coord[-1],
                'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputs_class[:-1], outputs_coord[:-1])]}

    '''
    def forward_return_attention(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        attention = self.backbone(samples.tensors, return_attention=True)
        return attention
    '''


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


@hydra.main(config_path="C:\\Users\pietr\PycharmProjects\idus\config", config_name="config")
def main(cfg: DictConfig):
    # Set up device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Yolos(cfg).to(device)
    post_process = PostProcess()

    # Example: Create a dummy input tensor
    dummy_input = torch.randn(cfg.batchSize, cfg.inChans, cfg.targetHeight, cfg.targetWidth).to(device)

    # Run model
    outputs = model(dummy_input)

    # Assert the output structure
    assert isinstance(outputs, dict), "Model output is not a dictionary"
    assert 'class' in outputs, "Key 'class' missing in model output"
    assert 'bbox' in outputs, "Key 'bbox' missing in model output"

    # Check dimensions
    batch_size = cfg.batchSize
    num_classes = cfg.numClass + 1  # Including background class
    assert outputs['class'].shape[0] == cfg.batchSize, \
        f"Expected batch size {cfg.batchSize} in 'class', got {outputs['class'].shape[0]}"
    assert outputs['class'].shape[1] == 100, \
        f"Expected 100 detection tokens in 'class', got {outputs['class'].shape[1]}"
    assert outputs['class'].shape[2] == cfg.numClass + 1, \
        f"Expected {cfg.numClass + 1} classes in 'class', got {outputs['class'].shape[2]}"

    assert outputs['bbox'].shape[0] == cfg.batchSize, \
        f"Expected batch size {cfg.batchSize} in 'bbox', got {outputs['bbox'].shape[0]}"
    assert outputs['bbox'].shape[1] == 100, \
        f"Expected 100 detection tokens in 'bbox', got {outputs['bbox'].shape[1]}"
    assert outputs['bbox'].shape[2] == 4, \
        f"Expected 4 bounding box coordinates in 'bbox', got {outputs['bbox'].shape[2]}"

    # Target sizes (for post-processing)
    target_sizes = torch.tensor([[cfg.targetHeight, cfg.targetWidth]] * cfg.batchSize, device=device)

    # Post-process results
    results = post_process(outputs, target_sizes)

    results = post_process(outputs, target_sizes)

    # Assert the structure of post-processed results
    assert isinstance(results, list), "Post-processed results should be a list"
    assert len(results) == cfg.batchSize, f"Expected {cfg.batchSize} results, got {len(results)}"
    for i, result in enumerate(results):
        assert 'scores' in result, f"Missing 'scores' in results[{i}]"
        assert 'labels' in result, f"Missing 'labels' in results[{i}]"
        assert 'boxes' in result, f"Missing 'boxes' in results[{i}]"
        assert result['boxes'].shape[-1] == 4, "Bounding box does not have 4 coordinates"

    print("All assertions passed!")

    # Optional: Save model weights
    if cfg.weight:
        os.makedirs(os.path.dirname(cfg.weight), exist_ok=True)
        torch.save(model.state_dict(), cfg.weight)
        print(f"Model weights saved to {cfg.weight}")


if __name__ == "__main__":
    main()

