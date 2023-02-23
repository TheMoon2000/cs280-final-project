import torch
import torch.nn as nn
import tqdm
from model import *
from utils.data import *
from utils import metrics
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    train_loader = DataLoader(RealBlurDataset(), batch_size=3, shuffle=True, pin_memory=True)
    val_loader = DataLoader(RealBlurDataset(train=False), batch_size=1, pin_memory=True)

    model = MainNet().to(device)
    evaluate(model, val_loader) # Get starting stats

    loss_fn = CharbonnierLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)

    for epoch in range(3):
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
            if len(epoch_losses) > 150: break
        print(f"Average training loss: {np.round(np.mean(epoch_losses), 4)}")

        evaluate(model, val_loader)

def evaluate(model: MainNet, val_loader: DataLoader):
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
                metrics.calculate_psnr(
                    metrics.tensor2uint(pred[i]),
                    metrics.tensor2uint(gt[i]),
                    input_order='HWC'
                ),
            )
            ssim.append(
                metrics.calculate_ssim(
                    metrics.tensor2uint(pred[i]),
                    metrics.tensor2uint(gt[i]),
                    input_order='HWC'
                ),
            )
        if len(psnr) > 100: break # temporarily limit eval size
    print(f"Average loss: {np.round(np.mean(losses), 4)}")
    print(f"Average PSNR: {np.round(np.mean(psnr), 4)}; Average SSIM: {np.round(np.mean(ssim), 4)}")


if __name__ == '__main__':
    main()