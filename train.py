#!/usr/bin/env python3

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from image_folder import *
from models.model_ae_conv_32x32x32_bin import AutoencoderConv
from utils import save_imgs


def train(args):
    model = AutoencoderConv().cuda()
    if args.load:
        model.load_state_dict(torch.load(args.chkpt))
        print("Loaded model from", args.chkpt)
    model.train()
    print("Done setup model")

    dataset = ImageFolder1024sqr(args.dataset_path)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers
    )
    print("Done setup dataloader: {len} batches of size {batch_size}".format(len=len(dataloader), batch_size=args.batch_size))

    mse_loss = nn.MSELoss()
    adam = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    sgd = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    optimizer = adam

    for ei in range(args.resume_epoch, args.num_epochs):
        for bi, (img, patches, _) in enumerate(dataloader):
            w = patches.shape[3]
            h = patches.shape[2]
            avg_loss = 0
            for i in range(h):
                for j in range(w):
                    x = Variable(patches[:, :, i, j, :, :]).cuda()
                    y = model(x)
                    loss = mse_loss(y, x)
                    avg_loss += loss.item() / len(dataset)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print('[%3d/%3d][%5d/%5d] loss: %f' % (ei, args.num_epochs, bi, len(dataloader), avg_loss))

            # save img
            if bi % args.out_every == 0:
                out = torch.zeros(h, w, 3, 128, 128)
                for i in range(h):
                    for j in range(w):
                        x = Variable(patches[0, :, i, j, :, :].unsqueeze(0)).cuda()
                        out[i, j] = model(x).cpu().data
                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (h*128, w*128, 3))
                out = np.transpose(out, (2, 0, 1))
                y = torch.cat((img[0], out), dim=2).unsqueeze(0)
                save_imgs(imgs=y, to_size=(3, h*128, 2 * w*128), name="out/{exp_name}/out_{ei}_{bi}.png".format(exp_name=args.exp_name, ei=ei, bi=bi))

            # save model
            if bi % args.save_every == args.save_every - 1:
                torch.save(model.state_dict(), "checkpoints/{exp_name}/model_{ei}_{bi}.state".format(exp_name=args.exp_name, ei=ei, bi=bi))

    torch.save(model.state_dict(), "checkpoints/{exp_name}/model_final.state".format(exp_name=args.exp_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--chkpt', type=str)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--out_every', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    os.makedirs("out/{exp_name}".format(exp_name=args.exp_name), exist_ok=True)
    os.makedirs("checkpoints/{exp_name}".format(exp_name=args.exp_name), exist_ok=True)
    train(args)
