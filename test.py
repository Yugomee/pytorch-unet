import torch
import torchvision
import os
import yaml
import argparse
from datasets import Cityscapes
from model import UNet
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

#From yoojin
def save_preds(mask, gt, save_path):
    colors = np.array([
        [255, 0, 0],    # Class 0
        [0, 255, 0],    # Class 1
        [0, 0, 255],    # Class 2
        [255, 255, 0]   # Class 3
    ], dtype=np.uint8)

    #Jeehyun : Assuming all inference use batch size 1
    mask = mask[0]

    color_mask = colors[mask]
    color_mask = Image.fromarray(color_mask)

    gt = TF.to_pil_image(gt[0])
    concat_img = Image.new('RGB', (gt.width * 2, gt.height))
    concat_img.paste(gt, (0, 0))
    concat_img.paste(color_mask, (gt.width, 0))
    concat_img.save(save_path, 'png')


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.configs) as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)

    test_config = configs['test']

    val_dataset = Cityscapes(data_dir=test_config['data_dir'], resize=tuple(test_config['image_size']), split='test')
    val_loader = DataLoader(val_dataset, batch_size=test_config['batch_size'], shuffle=False)

    #Load model
    model = UNet(in_dim=3, num_filters=64, num_classes=test_config['num_classes'])
    model_state = torch.load(test_config['model_path'])
    model.load_state_dict(model_state)
    model.to(device)

    #save path
    save_path = test_config['save_path']
    os.makedirs(save_path, exist_ok=True)

    for batch, (images, labels) in enumerate(tqdm(val_loader, desc='Inference...')):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).cpu()
            preds = F.interpolate(preds, size = (1024, 2048), mode='bilinear')
            images = F.interpolate(images, size=(1024, 2048), mode ='bilinear')
            pred_masks = torch.argmax(preds, dim=1).numpy()

            save_dir = os.path.join(save_path, f'pred_{batch}.png')
            save_preds(pred_masks, images, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get configs ")
    parser.add_argument('--configs', type=str, default='configs.yaml', help='Config path')
    args = parser.parse_args()
    main(args)


