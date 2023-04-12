import os
import torch
import torch.nn as nn
import tqdm
from model import *
from dataloaders.RealBlur import *
from utils import utils
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(model_name: str, epochs: int, dropout: float):
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    print(f"Training model {model_name}...")
    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])
    train_loader = DataLoader(RealBlurDataset(augmentation=aug), batch_size=5, shuffle=True, pin_memory=True)
    val_loader = DataLoader(RealBlurDataset(train=False), batch_size=5, shuffle=False, pin_memory=True)

    model = UformerSimple(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
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

    loss_fn = CharbonnierLoss()

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
        eval_loss, eval_psnr, eval_ssim = evaluate(model, val_loader)
        stats.append({
            'epoch': epoch + 1,
            'train_loss': np.mean(epoch_losses),
            'eval_loss': eval_loss,
            'eval_psnr': eval_psnr,
            'eval_ssim': eval_ssim
        })
        torch.save(model.state_dict(), model_name + '/model.pt')
        torch.save(optimizer.state_dict(), model_name + '/optimizer.pt')
        with open(model_name + '/stats.pk', 'wb') as f:
            pickle.dump(stats, f)

def evaluate(model: UformerSimple, val_loader: DataLoader, full=True):
    print("Evaluating model...")
    model.eval()
    loss_fn = CharbonnierLoss()
    losses = []
    psnr = []
    ssim = []
    for blur, gt in tqdm.tqdm(val_loader):
        blur = blur.to(device)
        with torch.no_grad():
            pred = model(blur)
            losses.append(loss_fn(pred, gt.to(device)).cpu().item())
        
        for i in range(gt.shape[0]):
            psnr.append(
                utils.calculate_psnr(
                    utils.tensor2uint(pred[i]),
                    utils.tensor2uint(gt[i]),
                    input_order='HWC'
                ),
            )
            ssim.append(
                utils.calculate_ssim(
                    utils.tensor2uint(pred[i]),
                    utils.tensor2uint(gt[i]),
                    input_order='HWC'
                ),
            )
        if not full and len(psnr) > 50: break # temporarily limit eval size
    print(f"Average eval loss: {np.round(np.mean(losses), 4)}")
    print(f"Average PSNR: {np.round(np.mean(psnr), 4)}; Average SSIM: {np.round(np.mean(ssim), 4)}")
    return np.mean(losses), np.mean(psnr), np.mean(ssim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deblur trainer')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/basic')
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    main(args.model_name, args.epochs, args.dropout)