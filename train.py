import os
import torch
import torch.nn as nn
import tqdm
from models.uformer import *
from dataloaders.RealBlur import *
from utils import utils
from dataloaders.ImageNet import *
from models.dpt import DPT
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(task: str, arch: str, model_name: str, epochs: int):
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Load dataset
    train_dataset: Dataset
    eval_dataset: Dataset
    evaluation_fn: function
    if task == 'seg':
        train_dataset = ImageNet()
        evaluation_fn = seg_eval
    elif task == 'color':
        train_dataset = ...
        evaluation_fn = colorization_eval
    else:
        raise NotImplementedError()
    
    # Load model
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    if arch == 'dpt':
        arch = DPT(40)
    elif arch == 'axial':
        raise NotImplementedError()
    elif arch == 'unet':
        raise NotImplementedError()
    elif arch == 'uformer':
        model = UformerSimple(dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
        loss_fn = CharbonnierLoss()
    else:
        raise NotImplementedError()

    print(f"Training model {arch} on task {task}...")
    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, pin_memory=True)
    val_loader = DataLoader(eval_dataset, batch_size=5, shuffle=False, pin_memory=True)

    stats = []
    if os.path.exists(model_name + '/model.pt'):
        model.load_state_dict(torch.load(model_name + '/model.pt'))
        optimizer.load_state_dict(torch.load(model_name + '/optimizer.pt'))
        with open(model_name + '/stats.pk', 'rb') as f:
            stats = pickle.load(f)
        print(f"Resuming at epoch {len(stats)}")
    else:
        os.makedirs(model_name, exist_ok=True)
    # evaluate(model, val_loader, full=False) # Get starting stats


    for epoch in range(len(stats), epochs):
        print(f"Training epoch {epoch}...")
        model.train()
        epoch_losses = []
        for blur, gt in tqdm.tqdm(train_loader):
            pred = model(blur.to(device))
            loss = loss_fn(pred, gt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().item())
            # if len(epoch_losses) > 100: break
        print(f"Average training loss: {np.round(np.mean(epoch_losses), 4)}")
        eval_stats = evaluation_fn(model, val_loader)
        stats.append({
            'epoch': epoch + 1,
            'train_loss': np.mean(epoch_losses),
            **eval_stats
        })
        torch.save(model.state_dict(), model_name + '/model.pt')
        torch.save(optimizer.state_dict(), model_name + '/optimizer.pt')
        with open(model_name + '/stats.pk', 'wb') as f:
            pickle.dump(stats, f)


def seg_eval(model: nn.Module, val_loader: DataLoader):
    losses = []
    accuracies = []
    for img, label in tqdm.tqdm(val_loader):
        pred = model(img)
        accuracies.append(torch.mean(pred.argmax(dim=1) == label).cpu().item())
    
    return {
        "eval_loss": np.mean(losses),
        "top1_accuracy": np.mean(accuracies)
    }

def colorization_eval(model: nn.Module, val_loader: DataLoader):
    pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer architectures comparison')
    parser.add_argument('--task', type=str, default='seg', choices=['seg', 'color'])
    parser.add_argument('--arch', type=str, choices=['dpt', 'uformer', 'axial', 'unet'])
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--epochs', type=int, default=32)
    args = parser.parse_args()
    main(args.task, args.arch, args.checkpoint_path, args.epochs)