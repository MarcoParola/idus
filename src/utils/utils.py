import torch 
import os

def load_weights(model, weight_path, device):

    if weight_path:
        print(f'loading pre-trained weights from {weight_path}')
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print('no pre-trained weights found, training from scratch...')
    return model
        
def load_model(args):
    
    if args.model == 'detr':
        from src.models.detr.detr import DETR
        model = DETR(args)
    elif args.model == 'yolos':
        from src.models.yolos.yolos import Yolos
        model = Yolos(args)
    elif args.model == 'yolo':
        model = None
    else:
        raise ValueError(f'unknown model: {args.model}')

    if args.weight != '':
        device = torch.device(args.device)
        model_path = os.path.join(args.currentDir, args.weight)
        model = load_weights(model, model_path, device)

    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    return model