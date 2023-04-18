import os
import torch
import torch.nn as nn
import tqdm
from models.uformer import *
from dataloaders.RealBlur import *
from dataloaders.ColorizationDataset import *
from models.dpt import DPT
import argparse
from utils import calculate_fid_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.axialtransformer import AxialImageTransformer
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(task: str, arch: str, model_name: str, epochs: int, batch_size: int):
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Load dataset
    train_dataset: Dataset
    eval_dataset: Dataset
    evaluation_fn: function
    loss_fn: nn.Module
    if task == 'seg':
        train_dataset = ...
        evaluation_fn = seg_eval
    elif task == 'color':
        train_dataset = ColorizationDataset(train=True)
        eval_dataset = ColorizationDataset(train=False)
        evaluation_fn = colorization_eval
        loss_fn = lambda pred, gt: torch.sum((pred - gt) ** 2, dim=1).mean()
    else:
        raise NotImplementedError()
    
    # Load model
    model: nn.Module
    optimizer: torch.optim.Optimizer
    if arch == 'dpt':
        model = DPT(40 if task == 'seg' else 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    elif arch == 'axial':
        model = AxialImageTransformer(dim=128, depth=12, reversible=True)
    elif arch == 'unet':
        raise NotImplementedError()
    elif arch == 'uformer':
        model = UformerSimple(dropout=0.1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
        loss_fn = CharbonnierLoss()
    else:
        raise NotImplementedError()
    model = model.to(device)

    print(f"Training model {arch} on task {task}...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

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
        for img, gt in tqdm.tqdm(train_loader):
            img = img.to(device)
            gt = gt.to(device)
            pred = model(img.to(device))
            loss = loss_fn(pred, gt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().item())
            # if len(epoch_losses) > 100: break
        print(f"Average training loss: {np.round(np.mean(epoch_losses), 4)}")
        torch.save(model.state_dict(), model_name + '/model.pt')
        eval_stats = evaluation_fn(model, val_loader, loss_fn)
        stats.append({
            'epoch': epoch + 1,
            'train_loss': np.mean(epoch_losses),
            **eval_stats
        })
        print(f"Average eval loss: {np.round(eval_stats['eval_loss'], 4)}")
        torch.save(optimizer.state_dict(), model_name + '/optimizer.pt')
        with open(model_name + '/stats.pk', 'wb') as f:
            pickle.dump(stats, f)


def seg_eval(model: nn.Module, val_loader: DataLoader):
    losses = []
    accuracies = []
    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        accuracies.append(torch.mean(pred.argmax(dim=1) == label).cpu().item())
    
    return {
        "eval_loss": np.mean(losses),
        "top1_accuracy": np.mean(accuracies)
    }

def colorization_eval(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module):
    losses = []
    fids = []
    model.eval()
    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(img)
            loss = loss_fn(pred, label)
        losses.append(loss.cpu().item())
        # fid = calculate_fid_score(pred, label)
        # fids.append(fid)
    
    return {
        "eval_loss": np.mean(losses),
        # "fid": np.mean(fids)
    } 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer architectures comparison')
    parser.add_argument('--task', type=str, default='seg', choices=['seg', 'color'])
    parser.add_argument('--arch', type=str, choices=['dpt', 'uformer', 'axial', 'unet'])
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    args = parser.parse_args()
    main(args.task, args.arch, args.checkpoint_path, args.epochs, args.batch_size)