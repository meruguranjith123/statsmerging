# import os
# import torch
# import torch.nn as nn
# import logging
# from task_vectors import TaskVector
# from eval import eval_single_dataset_preprocess_head
# from args import parse_arguments
# from heads import get_classification_head
# from datasets.registry import get_dataset
# from datasets.common import get_dataloader_shuffle, maybe_dictionarize

# # Define the model classes with complete implementation
# class ModelWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         if hasattr(self.model, 'transformer'):
#             delattr(self.model, 'transformer')
#     def forward(self, images):
#         return self.model(images)

# class LayerwiseAlphaPredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_additional_tasks, num_layers):
#         super().__init__()
#         self.num_additional_tasks = num_additional_tasks
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             layer = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, num_additional_tasks),
#                 nn.Softmax(dim=-1)
#             )
#             self.layers.append(layer)

#     def forward(self, x):
#         # Returns alphas for additional tasks (sum to 1 for each layer)
#         return torch.stack([layer(x[i]) for i, layer in enumerate(self.layers)], dim=0)

# class AdaMerging(nn.Module):
#     def __init__(self, paramslist, model, names, exam_datasets, args):
#         super().__init__()
#         self.paramslist = paramslist
#         self.model = model
#         self.names = names
#         self.exam_datasets = exam_datasets
#         self.args = args
#         num_layers = len(paramslist[0])
#         self.alpha_predictor = LayerwiseAlphaPredictor(
#             input_dim=6,
#             hidden_dim=256,
#             num_additional_tasks=len(paramslist)-1,
#             num_layers=num_layers
#         ).to(args.device)
        
#         # Initialize classification heads
#         for dataset_name in exam_datasets:
#             head = get_classification_head(args, dataset_name)
#             self.add_module(f'classifier_{dataset_name}', head.to(args.device))

#     def get_alphas(self):
#         layer_inputs = []
#         for layer_idx in range(len(self.paramslist[0])):
#             try:
#                 weights = [task[layer_idx] for task in self.paramslist[1:]]  
#                 stacked = torch.stack(weights, dim=0)
#                 flat = stacked.view(len(weights), -1).float()
                
#                 mean = flat.mean().unsqueeze(0)
#                 std = flat.std().unsqueeze(0)
#                 var = flat.var().unsqueeze(0)
#                 magnitude = flat.norm(p=2).unsqueeze(0)
#                 try:
#                     u, s, v = torch.svd(flat)
#                     top_sv = s[0].unsqueeze(0)
#                     rank = (s > 1e-3).float().sum().unsqueeze(0)
#                 except:
#                     top_sv = torch.zeros(1, device=self.args.device)
#                     rank = torch.zeros(1, device=self.args.device)
                
#                 stats = torch.cat([mean, std, var, magnitude, top_sv, rank], dim=0)
#             except:
#                 stats = torch.zeros(6, device=self.args.device)
#             layer_inputs.append(stats)
        
#         inputs = torch.stack(layer_inputs, dim=0)
#         additional_alphas = self.alpha_predictor(inputs.to(self.args.device))
#         full_alphas = torch.ones(len(self.paramslist[0]), len(self.paramslist), device=self.args.device)
#         full_alphas[:, 1:] = additional_alphas
#         return full_alphas.unsqueeze(0)

#     def get_classification_head(self, dataset_name):
#         return getattr(self, f'classifier_{dataset_name}')

#     def get_image_encoder(self):
#         alphas = self.get_alphas()[0]
#         merged_params = []
#         for l, params in enumerate(zip(*self.paramslist)):
#             weighted_param = params[0].clone()
#             for alpha, p in zip(alphas[l, 1:], params[1:]):
#                 weighted_param.add_(alpha * p)
#             merged_params.append(weighted_param)
#         load_weights(self.model, self.names, merged_params)
#         return self.model.to(self.args.device)

# def make_functional(mod):
#     orig_params, names = tuple(mod.parameters()), []
#     for name, _ in list(mod.named_parameters()):
#         del_attr(mod, name.split("."))
#         names.append(name)
#     return orig_params, names

# def del_attr(obj, names):
#     if len(names) == 1:
#         delattr(obj, names[0])
#     else:
#         del_attr(getattr(obj, names[0]), names[1:])

# def load_weights(mod, names, params):
#     for name, p in zip(names, params):
#         parts = name.split(".")
#         obj = mod
#         for n in parts[:-1]:
#             obj = getattr(obj, n)
#         setattr(obj, parts[-1], p)

# def create_log_dir(path, filename='log_inference.txt'):
#     os.makedirs(path, exist_ok=True)
#     logger = logging.getLogger(path)
#     logger.setLevel(logging.DEBUG)
#     if not logger.handlers:
#         fh = logging.FileHandler(os.path.join(path, filename))
#         fh.setLevel(logging.DEBUG)
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.DEBUG)
#         logger.addHandler(fh)
#         logger.addHandler(ch)
#     return logger

# def main():
#     # Configuration - should match your training setup
#     exam_datasets = ['Cars', 'EuroSAT', 'GTSRB', 'RESISC45']
#     model_name = 'ViT-B-32'
    
#     # Setup arguments
#     args = parse_arguments()
#     args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'
#     args.model = model_name
#     args.save = f'checkpoints-Stat-MLP-4model/{model_name}'
#     args.logs_path = f'logs-Stat-MLP-Stat-MLP-4model-Inference/{model_name}'
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Setup logging
#     log = create_log_dir(args.logs_path, 'log_inference.txt')
#     args.log = log
    
#     # Load pretrained model
#     pretrained_path = f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/zeroshot.pt'
#     pretrained_model = torch.load(pretrained_path)
#     model = ModelWrapper(pretrained_model).to(args.device)
#     orig_params, names = make_functional(model)
    
#     # Load task vectors
#     task_vectors = [TaskVector(
#         pretrained_path, 
#         f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/{d}/finetuned.pt'
#     ) for d in exam_datasets]
    
#     # Prepare parameters list
#     paramslist = [tuple(p.detach().to(args.device) for p in orig_params)]
#     paramslist += [tuple(p.detach().to(args.device) for p in tv.vector.values()) for tv in task_vectors]
    
#     # Initialize AdaMerging
#     adamerging = AdaMerging(paramslist, model, names, exam_datasets, args).to(args.device)
    
#     # Load trained weights
#     checkpoint_path = os.path.join(args.save, "best_model.pt")
#     if os.path.exists(checkpoint_path):
#         state_dict = torch.load(checkpoint_path, map_location=args.device)
#         adamerging.load_state_dict(state_dict)
#         log.info(f"Loaded trained model from {checkpoint_path}")
#     else:
#         raise FileNotFoundError(f"Could not find trained model at {checkpoint_path}")
    
#     # Run inference
#     adamerging.eval()
#     encoder = adamerging.get_image_encoder()
    
#     # Print alphas for inspection
#     alphas = adamerging.get_alphas()[0].detach().cpu()
#     log.info("\nLearned merging coefficients (alphas):")
#     log.info(f"Zeroshot weights: {alphas[:, 0].mean():.4f}")
#     for i, dataset in enumerate(exam_datasets):
#         log.info(f"{dataset} weights: mean={alphas[:, i+1].mean():.4f}, min={alphas[:, i+1].min():.4f}, max={alphas[:, i+1].max():.4f}")
    
#     # Evaluate on all datasets
#     log.info("\nEvaluation results:")
#     total_acc = 0.0
#     with torch.no_grad():
#         for d in exam_datasets:
#             classifier = adamerging.get_classification_head(d)
#             metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
#             total_acc += metrics['top1']
#             log.info(f"{d}: {metrics['top1']:.2f}%")
    
#     avg_acc = total_acc / len(exam_datasets)
#     log.info(f"\nAverage accuracy across all datasets: {avg_acc:.2f}%")

# if __name__ == "__main__":
#     main()


import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from PIL import Image
from task_vectors import TaskVector
from eval import eval_single_dataset_preprocess_head
from args import parse_arguments
from heads import get_classification_head
from datasets.registry import get_dataset
from datasets.common import get_dataloader_shuffle, maybe_dictionarize

# Define the model classes with attention visualization capability
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_maps = None
        if hasattr(self.model, 'transformer'):
            # Register hooks to capture attention maps
            self.register_hooks()
    
    def register_hooks(self):
        """Register hooks to capture attention maps from transformer layers"""
        def hook_fn(module, input, output):
            # output[1] contains the attention weights in ViT
            if self.attention_maps is None:
                self.attention_maps = []
            self.attention_maps.append(output[1].detach().cpu())
        
        for block in self.model.transformer.resblocks:
            block.attn.register_forward_hook(hook_fn)
    
    def forward(self, images):
        self.attention_maps = None  # Reset attention maps
        return self.model(images)
    
    def get_attention_maps(self):
        """Returns attention maps from all layers and heads"""
        if self.attention_maps is None:
            return None
        # Stack attention maps: [n_layers, batch_size, n_heads, seq_len, seq_len]
        return torch.stack(self.attention_maps)

class LayerwiseAlphaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_additional_tasks, num_layers):
        super().__init__()
        self.num_additional_tasks = num_additional_tasks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_additional_tasks),
                nn.Softmax(dim=-1)
            )
            self.layers.append(layer)

    def forward(self, x):
        # Returns alphas for additional tasks (sum to 1 for each layer)
        return torch.stack([layer(x[i]) for i, layer in enumerate(self.layers)], dim=0)

class AdaMerging(nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super().__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args
        num_layers = len(paramslist[0])
        self.alpha_predictor = LayerwiseAlphaPredictor(
            input_dim=6,
            hidden_dim=256,
            num_additional_tasks=len(paramslist)-1,
            num_layers=num_layers
        ).to(args.device)
        
        # Initialize classification heads
        for dataset_name in exam_datasets:
            head = get_classification_head(args, dataset_name)
            self.add_module(f'classifier_{dataset_name}', head.to(args.device))

    def get_alphas(self):
        layer_inputs = []
        for layer_idx in range(len(self.paramslist[0])):
            try:
                weights = [task[layer_idx] for task in self.paramslist[1:]]  
                stacked = torch.stack(weights, dim=0)
                flat = stacked.view(len(weights), -1).float()
                
                mean = flat.mean().unsqueeze(0)
                std = flat.std().unsqueeze(0)
                var = flat.var().unsqueeze(0)
                magnitude = flat.norm(p=2).unsqueeze(0)
                try:
                    u, s, v = torch.svd(flat)
                    top_sv = s[0].unsqueeze(0)
                    rank = (s > 1e-3).float().sum().unsqueeze(0)
                except:
                    top_sv = torch.zeros(1, device=self.args.device)
                    rank = torch.zeros(1, device=self.args.device)
                
                stats = torch.cat([mean, std, var, magnitude, top_sv, rank], dim=0)
            except:
                stats = torch.zeros(6, device=self.args.device)
            layer_inputs.append(stats)
        
        inputs = torch.stack(layer_inputs, dim=0)
        additional_alphas = self.alpha_predictor(inputs.to(self.args.device))
        full_alphas = torch.ones(len(self.paramslist[0]), len(self.paramslist), device=self.args.device)
        full_alphas[:, 1:] = additional_alphas
        return full_alphas.unsqueeze(0)

    def get_classification_head(self, dataset_name):
        return getattr(self, f'classifier_{dataset_name}')

    def get_image_encoder(self):
        alphas = self.get_alphas()[0]
        merged_params = []
        for l, params in enumerate(zip(*self.paramslist)):
            weighted_param = params[0].clone()
            for alpha, p in zip(alphas[l, 1:], params[1:]):
                weighted_param.add_(alpha * p)
            merged_params.append(weighted_param)
        load_weights(self.model, self.names, merged_params)
        return self.model.to(self.args.device)

def make_functional(mod):
    orig_params, names = tuple(mod.parameters()), []
    for name, _ in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        parts = name.split(".")
        obj = mod
        for n in parts[:-1]:
            obj = getattr(obj, n)
        setattr(obj, parts[-1], p)

def create_log_dir(path, filename='log_inference.txt'):
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(path, filename))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def visualize_attention(model_wrapper, image, save_path=None):
    """
    Visualize attention maps from the ViT model
    Args:
        model_wrapper: ModelWrapper instance
        image: Input image tensor (preprocessed)
        save_path: Optional path to save visualization
    """
    # Forward pass to get attention maps
    with torch.no_grad():
        _ = _ = model_wrapper(image.unsqueeze(0).to(next(model_wrapper.parameters()).device))
    
    # Get attention maps
    attention_maps = model_wrapper.get_attention_maps()
    if attention_maps is None:
        print("No attention maps found!")
        return
    
    # Process attention maps
    num_layers = attention_maps.shape[0]
    num_heads = attention_maps.shape[2]
    
    # Get the CLS token attention for each head in last layer
    cls_attention = attention_maps[-1, 0, :, 0, 1:]  # [n_heads, n_patches]
    
    # Average attention across heads
    avg_attention = cls_attention.mean(0)
    
    # Reshape to 2D grid (assuming square image patches)
    grid_size = int(np.sqrt(avg_attention.shape[-1]))
    attention_grid = avg_attention.reshape(grid_size, grid_size)
    
    # Interpolate to original image size
    orig_image = image.permute(1, 2, 0).cpu().numpy()
    orig_image = (orig_image - orig_image.min()) / (orig_image.max() - orig_image.min())
    h, w = orig_image.shape[:2]
    
    attention_img = torch.nn.functional.interpolate(
        attention_grid.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear'
    ).squeeze().numpy()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(orig_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    im = ax2.imshow(attention_img, cmap='hot')
    ax2.set_title('Attention Heatmap')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # Configuration
    exam_datasets = ['Cars', 'EuroSAT', 'GTSRB', 'RESISC45']
    model_name = 'ViT-B-32'
    
    # Setup arguments
    args = parse_arguments()
    args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'
    args.model = model_name
    args.save = f'checkpoints-Stat-MLP-4model/{model_name}'
    args.logs_path = f'logs-Stat-MLP-4model-Visual/{model_name}'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    log = create_log_dir(args.logs_path, 'log_inference.txt')
    args.log = log
    
    # Load pretrained model
    pretrained_path = f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/zeroshot.pt'
    pretrained_model = torch.load(pretrained_path)
    model = ModelWrapper(pretrained_model).to(args.device)
    orig_params, names = make_functional(model)
    
    # Load task vectors
    task_vectors = [TaskVector(
        pretrained_path, 
        f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/{d}/finetuned.pt'
    ) for d in exam_datasets]
    
    # Prepare parameters list
    paramslist = [tuple(p.detach().to(args.device) for p in orig_params)]
    paramslist += [tuple(p.detach().to(args.device) for p in tv.vector.values()) for tv in task_vectors]
    
    # Initialize AdaMerging
    adamerging = AdaMerging(paramslist, model, names, exam_datasets, args).to(args.device)
    
    # Load trained weights
    checkpoint_path = os.path.join(args.save, "best_model.pt")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        adamerging.load_state_dict(state_dict)
        log.info(f"Loaded trained model from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Could not find trained model at {checkpoint_path}")
    
    # Run inference
    adamerging.eval()
    adamerging.get_image_encoder()
    model.eval()
    
    # Print alphas
    alphas = adamerging.get_alphas()[0].detach().cpu()
    log.info("\nLearned merging coefficients (alphas):")
    log.info(f"Zeroshot weights: {alphas[:, 0].mean():.4f}")
    for i, dataset in enumerate(exam_datasets):
        log.info(f"{dataset} weights: mean={alphas[:, i+1].mean():.4f}, min={alphas[:, i+1].min():.4f}, max={alphas[:, i+1].max():.4f}")
    
    # Evaluate on all datasets
    # log.info("\nEvaluation results:")
    # total_acc = 0.0
    # with torch.no_grad():
    #     for d in exam_datasets:
    #         classifier = adamerging.get_classification_head(d)
    #         metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
    #         total_acc += metrics['top1']
    #         log.info(f"{d}: {metrics['top1']:.2f}%")
    
    # avg_acc = total_acc / len(exam_datasets)
    # log.info(f"\nAverage accuracy across all datasets: {avg_acc:.2f}%")
    
    # Visualization: Get sample images and visualize attention
    log.info("\nVisualizing attention maps...")
    for d in exam_datasets:
        dataset = get_dataset(d, pretrained_model.val_preprocess, location=args.data_location, batch_size=1)
        dataloader = get_dataloader_shuffle(dataset)
        
        # Get one sample image
        sample = next(iter(dataloader))
        sample = maybe_dictionarize(sample)
        image, label = sample['images'][0], sample['labels'][0]
        
        # Visualize attention
        save_path = os.path.join(args.logs_path, f"attention_{d}.png")
        visualize_attention(model, image, save_path)
        log.info(f"Saved attention visualization for {d} at {save_path}")

if __name__ == "__main__":
    main()