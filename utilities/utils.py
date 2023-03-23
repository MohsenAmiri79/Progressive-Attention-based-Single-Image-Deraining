from lightning.pytorch.callbacks import ProgressBar
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

import torch

from utilities.evaluation import SSIM


def mkdict(state_dict, validation_losses, epoch, optimizer, scheduler=None):
    out = {
        'state_dict': state_dict,
        'validation_losses': validation_losses,
        'epoch': epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    return out


def plot_loss(break_point, Valid_Loss_list, learning_rate_list, Loss_list):
    plt.figure(dpi=500)

    plt.subplot(211)
    x = range(break_point)
    y = Loss_list
    plt.plot(x, y, 'ro-', label='Train Loss')
    plt.plot(range(break_point), Valid_Loss_list, 'bs-', label='Valid Loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')

    plt.subplot(212)
    plt.plot(x, learning_rate_list, 'ro-', label='Learning rate')
    plt.ylabel('Learning rate')
    plt.xlabel('epochs')

    plt.legend()
    plt.show()


def train_model(model_list, train_loader, valid_loader, device, optimizer=None, scheduler=None, epoch=100, epoch_s=0, lr=1e-3, patience=30):
    model_name = model_list[1]
    model = model_list[0]
    stale = 0
    best_valid_loss = 10000
    break_point = 0
    if not optimizer: optimizer = optim.RAdam(model.parameters(), lr=lr)
    if not scheduler: scheduler = CosineAnnealingLR(optimizer, T_max=epoch+epoch_s)

    criterion = SSIM()

    Loss_list = []
    Valid_Loss_list = []
    learning_rate_list = []

    for i in range(epoch_s, epoch):
    # ---------------Train----------------
        model.train()
        train_losses = []
        
        for _, batch in enumerate(tqdm(train_loader)):
            inputs, labels = batch
            
            outputs = model(inputs.to(device))

            loss = - criterion(labels.to(device), outputs)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            train_losses.append(loss.item())
            
        train_loss = sum(train_losses) / len(train_losses)
        Loss_list.append(train_loss)
        print(f"[ Train | {i + 1:03d}/{epoch:03d} ] SSIM_loss = {train_loss:.5f}")
        
        scheduler.step()
        for param_group in optimizer.param_groups:
            learning_rate_list.append(param_group["lr"])
            print('learning rate %f' % param_group["lr"])
        
    # -------------Validation-------------
        model.eval()
        valid_losses = []
        for batch in tqdm(valid_loader):
            inputs, labels = batch

            with torch.no_grad():
                outputs = model(inputs.to(device))
            loss = criterion(labels.to(device), outputs)
            loss = -loss
            
            valid_losses.append(loss.item())
        
        valid_loss = sum(valid_losses) / len(valid_losses)
        Valid_Loss_list.append(valid_loss)
        print(f"[ Valid | {i + 1:03d}/{epoch:03d} ] SSIM_loss = {valid_loss:.5f}")
        
        break_point = i + 1

        if valid_loss < best_valid_loss:
            print(
                f"[ Valid | {i + 1:03d}/{epoch:03d} ] SSIM_loss = {valid_loss:.5f} -> best")
            print(f'Best model found at epoch {i+1}, saving model')

            net = mkdict(model.state_dict(), Valid_Loss_list, epoch, optimizer, scheduler)
            torch.save(net, f'trained_models/{model_name}/Low Data/best_model.pth')

            best_valid_loss = valid_loss
            stale = 0
        else:
            print(
                f"[ Valid | {i + 1:03d}/{epoch:03d} ] SSIM_loss = {valid_loss:.5f}")
            stale += 1
            if stale > patience:
                print(f'No improvement {patience} consecutive epochs, early stopping.')
                break

        net = mkdict(model.state_dict(), Valid_Loss_list, epoch, optimizer, scheduler)
        torch.save(net, f'trained_models/{model_name}/Low Data/last_model.pth')

    result = {
        'break_point':break_point,
        'Valid_Loss_list':Valid_Loss_list,
        'learning_rate_list':learning_rate_list,
        'Loss_list':Loss_list,
    }

    return result


def train_lightning(model, train_loader, val_loader, epochs=100):
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[ProgressBar()])
    trainer.fit(model, train_loader, val_loader)