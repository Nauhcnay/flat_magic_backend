import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision import utils
from PIL import Image
from io import BytesIO

dir_line = './flatting/size_org/line'
dir_edge = './flatting/size_org/line_detection'
dir_checkpoint = './checkpoints'

def denormalize(img):
    # denormalize
    inv_normalize = T.Normalize( mean=[-1], std=[2])

    img_np = inv_normalize(img)
    img_np = img_np.clamp(0, 1)
    # to numpy
    return img_np

def train_net(net,
              device,
              epochs=100,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              crop_size = None):

    # dataset = BasicDataset(dir_line, dir_edge, crop_size = crop_size)
    logging.info("Loading training set to memory")
    lines_bytes, edges_bytes = load_to_ram(dir_line, dir_edge)

    dataset = BasicDataset(lines_bytes, edges_bytes, crop_size = crop_size)
    
    n_train = len(dataset)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    # we don't need valiation currently
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    
    global_step = 0

    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}
    #     Batch size:      {batch_size}
    #     Learning rate:   {lr}
    #     Training size:   {n_train}
    #     Validation size: {n_val}
    #     Checkpoints:     {save_cp}
    #     Device:          {device.type}
    #     Images scaling:  {img_scale}
    # ''')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Crop size:       {crop_size}
    ''')

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    # how to use scheduler?
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    
    # since now we are trying to generate images, so we use l1 loss
    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()

    criterion = nn.L1Loss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, gts, mask1, mask2 in train_loader:

                # assert imgs.shape[1] == net.in_channels, \
                #     f'Network has been defined with {net.in_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)

                # forward
                pred = net(imgs)
                
                '''
                baseline
                '''
                # loss1 = criterion(pred, gts)
                
                '''
                weighted loss
                '''
                mask_1 = (1-mask1)
                mask_2 = 100 * (1-mask2)
                mask_3 = 0.5 * mask2
                mask_w = mask_1 + mask_2 + mask_3
                mask_w = mask_w.to(device=device, dtype=torch.float32)
                loss1 = criterion(pred * mask_w, gts * mask_w)
                
                '''
                point number loss
                the point number of the perdiction and gt should close, too
                
                '''
                loss2 = criterion(
                        ((denormalize(gts)==0).sum()).float(),
                        ((denormalize(pred)==0).sum()).float()
                        )
                
                # total loss
                loss = loss1 + 0.5 * torch.log(torch.abs(loss2 + 1))

                epoch_loss += loss.item()
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/l1', loss1.item(), global_step)
                writer.add_scalar('Loss/point', loss2.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # back propagate
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                
                global_step += 1

                # if global_step % (n_train // (10 * batch_size)) == 0:
                #     for tag, value in net.named_parameters():
                #         tag = tag.replace('.', '/')
                #         writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #         writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                #     val_score = eval_net(net, val_loader, device)
                #     scheduler.step(val_score)
                #     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                #     if net.n_classes > 1:
                #         logging.info('Validation cross entropy: {}'.format(val_score))
                #         writer.add_scalar('Loss/test', val_score, global_step)
                #     else:
                #         logging.info('Validation Dice Coeff: {}'.format(val_score))
                #         writer.add_scalar('Dice/test', val_score, global_step)

                #     writer.add_images('images', imgs, global_step)
                #     if net.n_classes == 1:
                #         writer.add_images('masks/true', true_masks, global_step)
                #         writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                
                if global_step % 1000 == 0:
                    sample = torch.cat((imgs, pred, gts), dim = 0)
                    if os.path.exists("./results/train/") is False:
                        logging.info("Creating ./results/train/")
                        os.makedirs("./results/train/")

                    utils.save_image(
                        sample,
                        f"./results/train/{str(global_step).zfill(6)}.png",
                        nrow=int(batch_size),
                        # nrow=int(sample.shape[0] ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

        if save_cp and epoch % 100 == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'/CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def save_to_ram(path_to_img):

    img = Image.open(path_to_img).convert("L")
    buffer = BytesIO()
    img.save(buffer, format='png')

    return buffer.getvalue()

def load_to_ram(path_to_line, path_to_edge):
    lines = os.listdir(path_to_line)
    lines.sort()

    edges = os.listdir(path_to_edge)
    edges.sort()

    assert len(lines) == len(edges)

    lines_bytes = []
    edges_bytes = []

    # read everything into memory
    for img in tqdm(lines):
        assert img.replace("webp", "png") in edges
        
        lines_bytes.append(save_to_ram(os.path.join(path_to_line, img)))
        edges_bytes.append(save_to_ram(os.path.join(path_to_edge, img.replace("webp", "png"))))

    return lines_bytes, edges_bytes


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=90000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-m', '--multi-gpu', action='store_true')
    parser.add_argument('-c', '--crop-size', metavar='C', type=int, default=1024,
                        help='the size of random cropping')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    
    __spec__ = None
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    net = UNet(in_channels=1, out_channels=1, bilinear=True)
    
    if args.multi_gpu:
        logging.info("using data parallel")
        net = nn.DataParallel(net).cuda()
    else:
        net.to(device=device)

    # logging.info(f'Network:\n'
    #              f'\t{net.in_channels} input channels\n'
    #              f'\t{net.out_channels} output channels\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    #              )

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  crop_size=args.crop_size)

    # this is interesting, save model when keyborad interrupt
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
        # logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
