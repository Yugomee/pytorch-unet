import torch
import os
import numpy as np
import yaml
import argparse
from datasets import Cityscapes
from model import UNet
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import time
from tqdm.auto import tqdm



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Get config
    with open(args.configs) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_config = config['train']
    wandb.init(project='unet', config=train_config)

    train_dataset = Cityscapes(data_dir=train_config['data_dir'], resize=tuple(train_config['image_size']), split='train')
    val_dataset = Cityscapes(data_dir=train_config['data_dir'], resize=tuple(train_config['image_size']), split='train')

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    model = UNet(in_dim=3, num_filters=64, num_classes=train_config['num_classes'])
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = train_config['learning_rate'])
    epochs = train_config['epochs']


    print('Train begins')
    for epoch in tqdm(range(epochs), desc = "Epochs"):
        model.train()
        total_train_loss = 0.0
        for batch, (images, labels) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch % train_config.get('log_interval', 10) == 0:
                wandb.log({'train_loss': loss.item(), 'epoch': epoch})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{train_config['epochs']}], Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images, labels = images.to(device), labels.to(device)

                preds = model(images)
                loss = criterion(preds, labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{train_config['epochs']}], Validation Loss: {avg_val_loss:.4f}")

        wandb.log({'val_loss': avg_val_loss, 'epoch': epoch})

        # Save model checkpoint
        if (epoch + 1) % train_config.get('save_interval', 5) == 0:
            checkpoint_path = os.path.join(train_config['checkpoint_path'], f'unet_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get configs ")
    parser.add_argument('--configs', type=str, default='configs.yaml', help='Config path')
    args = parser.parse_args()
    main(args)



