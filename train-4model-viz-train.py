#SUN397 Cars RESISC45 DTD SVHN GTSRB



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import time
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from task_vectors import TaskVector
# from eval import eval_single_dataset_preprocess_head
# from args import parse_arguments
# from heads import get_classification_head
# from datasets.registry import get_dataset
# from datasets.common import get_dataloader_shuffle, maybe_dictionarize

# # CUDA and memory tuning
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.max_split_size_mb = 128

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# class ModelWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         if hasattr(self.model, 'transformer'):
#             delattr(self.model, 'transformer')

#     def forward(self, images):
#         return self.model(images)

# class LayerwiseAlphaPredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_tasks, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, num_tasks),
#                 nn.Softmax(dim=-1)
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, x):
#         return torch.stack([layer(x) for layer in self.layers], dim=1)

# def make_functional(mod):
#     orig_params = tuple(mod.parameters())
#     names = []
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

# def create_log_dir(path, filename='log.txt'):
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

# class AdaMerging(nn.Module):
#     def __init__(self, paramslist, model, names, exam_datasets, args):
#         super().__init__()
#         self.paramslist = paramslist
#         self.model = model
#         self.names = names
#         self.exam_datasets = exam_datasets
#         self.args = args

#         num_layers = len(paramslist[0])
#         self.alpha_predictor = LayerwiseAlphaPredictor(6, 128, len(paramslist), num_layers).to(args.device)

#         for dataset_name in exam_datasets:
#             head = get_classification_head(args, dataset_name)
#             self.add_module(f'classifier_{dataset_name}', head.to(args.device))

#     def get_model_stats(self):
#         with torch.no_grad():
#             stds, vars, mean_vecs = [], [], []
#             magnitudes, ranks, top_sv = [], [], []

#             for i in range(len(self.paramslist)):
#                 layer_means = [p.mean() for p in self.paramslist[i]]
#                 mean_vecs.append(torch.stack(layer_means))

#             for layer_params in zip(*self.paramslist):
#                 stacked = torch.stack(layer_params, dim=0)
#                 flat = stacked.view(len(self.paramslist), -1)

#                 stds.append(stacked.std(dim=0).mean())
#                 vars.append(stacked.var(dim=0).mean())
#                 magnitudes.append(flat.abs().mean())

#                 try:
#                     u, s, v = torch.svd(flat)
#                     top_sv.append(s[0])
#                     ranks.append((s > 1e-3).float().sum())
#                 except RuntimeError:
#                     top_sv.append(torch.tensor(0., device=self.args.device))
#                     ranks.append(torch.tensor(0., device=self.args.device))

#             mean_vecs = torch.stack(mean_vecs)
#             cos_sim = F.cosine_similarity(mean_vecs[None, :], mean_vecs[:, None], dim=-1).mean()

#             return torch.tensor([
#                 torch.stack(stds).mean(),
#                 torch.stack(vars).mean(),
#                 cos_sim,
#                 torch.stack(magnitudes).mean(),
#                 torch.stack(top_sv).mean(),
#                 torch.stack(ranks).mean()
#             ], device=self.args.device).unsqueeze(0)

#     def get_alphas(self):
#         return self.alpha_predictor(self.get_model_stats())

#     def get_classification_head(self, dataset_name):
#         return getattr(self, f'classifier_{dataset_name}')

#     def get_image_encoder(self):
#         with torch.no_grad():
#             alphas = self.get_alphas()[0]
#             merged_params = []
#             for l, params in enumerate(zip(*self.paramslist)):
#                 weighted_param = torch.zeros_like(params[0])
#                 for alpha, p in zip(alphas[l], params):
#                     weighted_param.add_(alpha * p)
#                 merged_params.append(weighted_param)
#             load_weights(self.model, self.names, merged_params)
#             return self.model.to(self.args.device)

#     def forward(self, x, dataset_name, training=False):
#         alphas = self.get_alphas()[0] if training else self.get_alphas()[0].detach()
#         merged_params = []
#         for l, params in enumerate(zip(*self.paramslist)):
#             weighted_param = torch.zeros_like(params[0])
#             for alpha, p in zip(alphas[l], params):
#                 weighted_param.add_(alpha * p)
#             merged_params.append(weighted_param)
#         load_weights(self.model, self.names, merged_params)
#         features = self.model(x)
#         return self.get_classification_head(dataset_name)(features)

# def main():
#     exam_datasets = ['SUN397','EuroSAT', 'GTSRB', 'DTD', 'MNIST', 'Cars']
#     # SUN397 Cars GTSRB EuroSAT DTD MNIST
#     #SUN397 Cars RESISC45 DTD SVHN GTSRB
#     # exam_datasets =  ['Cars', 'RESISC45', 'EuroSAT','SUN397', 'SVHN', 'MNIST','GTSRB', 'DTD']
#     model_name = 'ViT-B-32'
#     args = parse_arguments()
#     args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'
#     args.model = model_name
#     args.save = f'checkpoints-Layer-wise-seen-task2/{model_name}'
#     args.logs_path = f'logs_plot-seen-task2/{model_name}'
#     args.device = device

#     log = create_log_dir(args.logs_path, f'log_{time.strftime("%Y%m%d_%H%M%S")}_AdaMerging.txt')
#     args.log = log

#     pretrained_path = f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/zeroshot.pt'
#     pretrained_model = torch.load(pretrained_path)
#     model = ModelWrapper(pretrained_model).to(args.device)
#     orig_params, names = make_functional(model)

#     task_vectors = [
#         TaskVector(pretrained_path, f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/{d}/finetuned.pt')
#         for d in exam_datasets
#     ]

#     paramslist = [tuple(p.detach().to(args.device).requires_grad_() for p in orig_params)]
#     paramslist += [tuple(p.detach().to(args.device).requires_grad_() for p in tv.vector.values()) for tv in task_vectors]
#     torch.cuda.empty_cache()

#     adamerging = AdaMerging(paramslist, model, names, exam_datasets, args).to(args.device)
#     optimizer = torch.optim.Adam(adamerging.parameters(), lr=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

#     # Initial eval (fast encoder reuse)
#     log.info("Initial evaluation:")
#     encoder = adamerging.get_image_encoder()
#     for d in exam_datasets:
#         classifier = adamerging.get_classification_head(d)
#         metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
#         log.info(f"{d} ACC: {metrics['top1']:.2f}")

#     best_avg_acc = -1
#     for epoch in range(4000):
#         start = time.time()
#         adamerging.train()
#         total_loss = torch.tensor(0., device=args.device)

#         for d in exam_datasets:
#             dataset = get_dataset(d, pretrained_model.val_preprocess, location=args.data_location, batch_size=16)
#             dataloader = get_dataloader_shuffle(dataset)
#             for batch_i, data in enumerate(dataloader):
#                 if batch_i >= 2: break
#                 data = maybe_dictionarize(data)
#                 x, y = data['images'].to(args.device), data['labels'].to(args.device)
#                 out = adamerging(x, d, training=True)
#                 total_loss += F.cross_entropy(out, y)

#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#         scheduler.step(total_loss)

#         log.info(f"Epoch {epoch} Loss: {total_loss.item():.4f} Time: {time.time()-start:.1f}s")

#         if (epoch + 1) % 100 == 0:
#             adamerging.eval()
#             encoder = adamerging.get_image_encoder()
#             total_acc = 0.
#             with torch.no_grad():
#                 for d in exam_datasets:
#                     classifier = adamerging.get_classification_head(d)
#                     metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
#                     total_acc += metrics['top1']
#                     log.info(f"{d}: {metrics['top1']:.2f}")
#             avg_acc = total_acc / len(exam_datasets)
#             log.info(f"Avg ACC: {avg_acc:.2f}\n")
#             if avg_acc > best_avg_acc:
#                 best_avg_acc = avg_acc
#                 torch.save(adamerging.state_dict(), args.save + "/best_model.pt")

# if __name__ == "__main__":
#     main()



#SUN397 Cars RESISC45 DTD SVHN GTSRB




# Improved AdaMerging with Stat-MLP
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_vectors import TaskVector
from eval import eval_single_dataset_preprocess_head
from args import parse_arguments
from heads import get_classification_head
from datasets.registry import get_dataset
from datasets.common import get_dataloader_shuffle, maybe_dictionarize

torch.backends.cudnn.benchmark = True
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.max_split_size_mb = 64


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
    def forward(self, images):
        return self.model(images)

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
                nn.Softmax(dim=-1)  # Ensure alphas sum to 1 for additional tasks
            )
            self.init_to_mid_range(layer)
            self.layers.append(layer)

    def init_to_mid_range(self, layer):
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.fill_(0.0)

    def forward(self, x):
        # Returns alphas for additional tasks (sum to 1 for each layer)
        return torch.stack([layer(x[i]) for i, layer in enumerate(self.layers)], dim=0)

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

def create_log_dir(path, filename='log.txt'):
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

class AdaMerging(nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super().__init__()
        self.paramslist = paramslist  # [zeroshot, task1, task2, ...]
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args
        num_layers = len(paramslist[0])
        self.alpha_predictor = LayerwiseAlphaPredictor(
            input_dim=6,
            hidden_dim=256,
            num_additional_tasks=len(paramslist)-1,  # Exclude zeroshot
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
                # Only consider additional tasks (exclude zeroshot)
                weights = [task[layer_idx] for task in self.paramslist[1:]]  
                stacked = torch.stack(weights, dim=0)
                flat = stacked.view(len(weights), -1).float()
                
                # Compute statistics
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
        
        # Get alphas for additional tasks (sum to 1 for each layer)
        additional_alphas = self.alpha_predictor(inputs.to(self.args.device))  # [L, n_tasks]
        
        # Create full alphas tensor [L, total_tasks] where total_tasks = n+1
        full_alphas = torch.ones(len(self.paramslist[0]), len(self.paramslist), device=self.args.device)
        full_alphas[:, 1:] = additional_alphas  # Set alphas for additional tasks
        
        return full_alphas.unsqueeze(0)  # Add batch dim

    def get_classification_head(self, dataset_name):
        return getattr(self, f'classifier_{dataset_name}')

    def get_image_encoder(self):
        alphas = self.get_alphas()[0]  # [L, T]
        merged_params = []
        for l, params in enumerate(zip(*self.paramslist)):
            # Start with zeroshot model (alpha=1)
            weighted_param = params[0].clone()
            
            # Add weighted contributions from additional models
            for alpha, p in zip(alphas[l, 1:], params[1:]):
                weighted_param.add_(alpha * p)
                
            merged_params.append(weighted_param)
            
        load_weights(self.model, self.names, merged_params)
        return self.model.to(self.args.device)

    def forward(self, x, dataset_name, training=False):
        alphas = self.get_alphas()[0]  # [L, T]
        merged_params = []
        for l, params in enumerate(zip(*self.paramslist)):
            weighted_param = params[0].clone()  # Zeroshot base
            for alpha, p in zip(alphas[l, 1:], params[1:]):  # Additional tasks
                weighted_param.add_(alpha * p)
            merged_params.append(weighted_param)
            
        load_weights(self.model, self.names, merged_params)
        features = self.model(x)
        return self.get_classification_head(dataset_name)(features)

def main():
    # exam_datasets = ['RESISC45', 'GTSRB', 'EuroSAT', 'Cars','SUN397','DTD','SVHN','MNIST']
    # exam_datasets = ['SUN397','RESISC45', 'GTSRB','DTD', 'SVHN', 'Cars']
    exam_datasets = ['Cars','EuroSAT', 'GTSRB','RESISC45']
    model_name = 'ViT-B-32'
    args = parse_arguments()
    args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'
    args.model = model_name
    args.save = f'checkpoints-Stat-MLP-task2/{model_name}'
    args.logs_path = f'logs-Stat-MLP-task2/{model_name}'


    #checkpoints-Stat-MLP-4model
    # args.device = device

    log = create_log_dir(args.logs_path, f'log_{time.strftime("%Y%m%d_%H%M%S")}_StatMLP.txt')
    args.log = log

    # Load models
    pretrained_path = f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/zeroshot.pt'
    pretrained_model = torch.load(pretrained_path)
    model = ModelWrapper(pretrained_model).to(args.device)
    orig_params, names = make_functional(model)

    # Load task vectors
    task_vectors = [TaskVector(
        pretrained_path, 
        f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/{d}/finetuned.pt'
    ) for d in exam_datasets]

    # Prepare parameters list: [zeroshot, task1, task2, ...]
    paramslist = [tuple(p.detach().to(args.device).requires_grad_() for p in orig_params)]
    paramslist += [tuple(p.detach().to(args.device).requires_grad_() for p in tv.vector.values()) for tv in task_vectors]
    torch.cuda.empty_cache()

    # Initialize AdaMerging
    adamerging = AdaMerging(paramslist, model, names, exam_datasets, args).to(args.device)
    optimizer = torch.optim.AdamW(adamerging.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Initial evaluation
    # log.info("Initial evaluation:")
    # encoder = adamerging.get_image_encoder()
    # log.info(f"Initial alphas:\n{adamerging.get_alphas()[0].detach().cpu()}")
    # for d in exam_datasets:
    #     classifier = adamerging.get_classification_head(d)
    #     metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
    #     log.info(f"{d} ACC: {metrics['top1']:.2f}")

    best_avg_acc = -1
    num_batches = 1  # Process more batches per epoch
    
        
    for epoch in range(5000):
        start = time.time()
        adamerging.train()
        total_loss = 0.0
        processed_batches = 0
        accumulated_loss = torch.tensor(0.0, device=args.device)

        for d in exam_datasets:
            dataset = get_dataset(d, pretrained_model.val_preprocess, location=args.data_location, batch_size=4)
            dataloader = get_dataloader_shuffle(dataset)

            for batch_i, data in enumerate(dataloader):
                if batch_i >= num_batches:
                    break

                data = maybe_dictionarize(data)
                x, y = data['images'].to(args.device), data['labels'].to(args.device)

                out = adamerging(x, d, training=True)
                loss = F.cross_entropy(out, y)

                accumulated_loss += loss
                total_loss += loss.item()
                processed_batches += 1

        optimizer.zero_grad()
        accumulated_loss.backward()
        optimizer.step()
        scheduler.step()

        avg_loss = total_loss / processed_batches
        log.info(f"Epoch {epoch} Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.2e} Time: {time.time()-start:.1f}s")

        # Evaluation every 50 epochs
        if (epoch + 1) % 50 == 0:
            adamerging.eval()
            encoder = adamerging.get_image_encoder()
            alphas = adamerging.get_alphas()[0].detach().cpu()
            log.info(f"Alphas at epoch {epoch + 1}:\nZeroshot weights: {alphas[:, 0].mean():.4f}\n"
                    f"Additional tasks weights:\n{alphas[:, 1:]}")

            total_acc = 0.0
            with torch.no_grad():
                for d in exam_datasets:
                    classifier = adamerging.get_classification_head(d)
                    metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
                    total_acc += metrics['top1']
                    log.info(f"{d}: {metrics['top1']:.5f}")

            avg_acc = total_acc / len(exam_datasets)
            log.info(f"Avg ACC: {avg_acc:.5f}\n")

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                torch.save(adamerging.state_dict(), args.save + "/best_model.pt")
                log.info(f"New best model saved with avg acc {best_avg_acc:.4f}")


if __name__ == "__main__":
    main()