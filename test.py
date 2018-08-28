#!/usr/bin/env python3

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from image_folder import ImageFolderAuto
from models.model_ae_conv_32x32x32_bin import AutoencoderConv
# latent size is 16x8x8 bits/patch (i.e. 3.75 bpp)
# from models.model_ae_conv_bin import AutoencoderConv
from utils import save_imgs


def test(args):
    model = AutoencoderConv().cuda()
    model.load_state_dict(torch.load(args.chkpt))
    model.eval()
    print("Loaded model from", args.chkpt)

    dataset = ImageFolderAuto(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=args.shuffle)
    print("Done setup dataloader: {len}".format(len=len(dataloader)))

    mse_loss = nn.MSELoss()

    for bi, (img, patches, path) in enumerate(dataloader):
        # print(patches.shape)
        w = patches.shape[3]
        h = patches.shape[2]
        out = torch.zeros(h, w, 3, 128, 128)
        # enc = torch.zeros(h, w, 16, 8, 8)
        avg_loss = 0
        
        for i in range(h):
            for j in range(w):
                x = Variable(patches[:, :, i, j, :, :]).cuda()
                y = model(x)
                # e = model.enc_x.data
                # p = torch.tensor(np.random.permutation(e.reshape(-1, 1)).reshape(1, 16, 8, 8)).cuda()
                # out[i, j] = model.decode(p).data
                # enc[i, j] = model.enc_x.data
                out[i, j] = y.data
                loss = mse_loss(y, x)
                avg_loss += loss.item() / len(dataset)

        print('[%5d/%5d] loss: %f' % (bi, len(dataloader), avg_loss))

        # save output
        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (h*128, w*128, 3))
        out = np.transpose(out, (2, 0, 1))

        #y = torch.cat((img[0], out), dim=2)
        save_imgs(imgs=out.unsqueeze(0), to_size=(3, h*128, w*128), name="./test/{out_dir}/{file_name}".format(out_dir=args.out_dir, file_name=os.path.basename(path[0])))


# save encoded
# enc = np.reshape(enc, -1)
# sz = str(len(enc)) + 'd'
# open(f"./{args.out_dir}/test_{bi}.enc", "wb").write(struct.pack(sz, *enc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./test_default')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs("./test/{out_dir}".format(out_dir=args.out_dir), exist_ok=True)

    test(args)
