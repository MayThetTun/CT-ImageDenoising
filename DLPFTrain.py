import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from dataset import Dataset
from models import FFDNet
from utils import weights_init_kaiming, batch_psnr, \
    svd_orthogonalization

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

psnrt = []
losst = []
lrs = []


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis <= r:
        return 1.0
    else:
        return 0


def mask_radial(img, r):
    bs, ch, rows, cols = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    mask = np.zeros((bs, ch, rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[:, :, i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def convertFreqImage(img):  # Convert Frequency Domain
    x = img.to('cpu').detach().numpy().copy()
    bs, c, M, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    r = 35
    H = mask_radial(np.zeros([bs, c, M, N]), r)
    H = np.fft.ifft2(H)
    TS = torch.Tensor(H)
    s = torch.cat((img, TS), 1)
    return s


def main(args):
    print('> Loading dataset ...')
    dataset_train = Dataset(train=True, gray_mode=args.gray, shuffle=True)
    dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=6,
                              batch_size=args.batch_size, shuffle=True)
    print("\t# of training samples: %d\n" % int(len(dataset_train)))

    # Init loggers
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create model
    if args.gray:
        in_ch = 1
    else:
        in_ch = 3
    net = FFDNet(num_input_channels=in_ch)

    # Initialize model
    net.apply(weights_init_kaiming)

    # Define loss Function
    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids)
    model.to('cuda')

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume training or start a new
    if args.resume_training:
        resumef = os.path.join(args.log_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = args.epochs
            args = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            args.epochs = new_epoch
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['args'])
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))
            args.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".
                            format(resumef))
    else:
        start_epoch = 0
        training_params = {'step': 0, 'current_lr': 0, 'no_orthog': args.no_orthog}

    sft = time.time()
    # Training
    best_psnr_val = 0.0
    for epoch in range(start_epoch, args.epochs):
        # train
        for i, data in enumerate(loader_train, 0):
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            gnoise = torch.zeros(img_train.size())
            pnoise = torch.zeros(img_train.size())
            imgn_train = torch.zeros(img_train.size())
            p_imgn_train = torch.zeros(img_train.size())
            gstdn = np.random.uniform(args.gnoiseIntL[0], args.gnoiseIntL[1],
                                      size=gnoise.size()[0])
            pstdn = np.random.uniform(args.pnoiseIntL[0], args.pnoiseIntL[1],
                                      size=pnoise.size()[0])
            for nx in range(gnoise.size()[0]):
                sizen = gnoise[0, :, :, :].size()
                gnoise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=gstdn[nx])

            for nx in range(pnoise.size()[0]):
                p_imgn_train[nx, :, :, :] = torch.poisson(img_train[nx, :, :, :] * pstdn[nx]) / pstdn[
                    nx]  # Poisson noisy image
                pnoise[nx, :, :, :] = p_imgn_train[nx, :, :, :] - img_train[nx, :, :, :]

            imgn_train = img_train + gnoise + pnoise
            noise = gnoise + pnoise
            stdn = gstdn + pstdn

            # convert
            imgn_train = convertFreqImage(imgn_train)
            img_train = img_train.to('cuda')
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))

            # Evaluate model and optimize it
            out_train = model(imgn_train.to('cuda'), stdn_var.to('cuda'))
            loss = criterion(out_train.to('cuda'), noise.to('cuda')) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            # Results
            model.eval()
            if args.gray:
                out_train = torch.clamp(
                    imgn_train[:, :1, :, :].to('cuda') - model(imgn_train.to('cuda'), stdn_var.to('cuda')),
                    0., 1.)
                psnr_train = batch_psnr(out_train, img_train, 1.)
            else:
                out_train = torch.clamp(
                    imgn_train[:, :3, :, :].to('cuda') - model(imgn_train.to('cuda'), stdn_var.to('cuda')),
                    0., 1.)
                psnr_train = batch_psnr(out_train, img_train, 1.)

            # Print results
            training_params['step'] += 1
            # The end of each epoch
            if training_params['step'] % args.save_every == 0:
                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % \
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

        lrs.append(optimizer.param_groups[0]["lr"])
        print("Learning rate", lrs)
        # scheduler.step()
        model.eval()
        psnrt.append(psnr_train)
        losst.append(loss.item())

        # Validation
        psnr_val = 0
        for valimg in dataset_val:
            img_val = torch.unsqueeze(valimg, 0)
            imgn_val = torch.zeros(img_val.size())
            gvnoise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=args.gval_noiseL)
            peak = args.pval_noiseL
            p_imgn_val = torch.poisson(img_val * peak) / peak  # poisson noise added
            pvnoise = p_imgn_val - img_val
            imgn_val = img_val + gvnoise + pvnoise
            val_noiseL = args.gval_noiseL + args.pval_noiseL
            # convert
            imgn_val = convertFreqImage(imgn_val)
            with torch.no_grad():
                img_val, imgn_val = img_val.to('cuda'), imgn_val.to('cuda')
                sigma_noise = torch.FloatTensor([val_noiseL]).to('cuda')
                if args.gray:
                    out_val = torch.clamp(
                        imgn_val[:, :1, :, :].to('cuda') - model(imgn_val.to('cuda'), sigma_noise.to('cuda')),
                        0., 1.)
                    psnr_val += batch_psnr(out_val, img_val, 1.)

                else:
                    out_val = torch.clamp(
                        imgn_val[:, :3, :, :].to('cuda') - model(imgn_val.to('cuda'), sigma_noise.to('cuda')),
                        0., 1.)
                    psnr_val += batch_psnr(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1

        torch.save(model.state_dict(), os.path.join(args.log_dir, 'net.pth'))

        # save model and checkpoint
        if psnr_val > best_psnr_val:
            best_psnr_val = psnr_val
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))

        save_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'training_params': training_params,
            'args': args
        }

        torch.save(save_dict, os.path.join(args.log_dir, 'ckpt.pth'))
        if epoch % args.save_every_epochs == 0:
            torch.save(save_dict, os.path.join(args.log_dir,
                                               'ckpt_e{}.pth'.format(epoch + 1)))
        del save_dict

    curr_time = (time.time() - sft)
    print("\nTest time on Training data: {0:.4f}s".format(curr_time))

    plt.plot(epoch)
    plt.xlabel('epoch')
    plt.plot(psnrt, '-o')
    plt.ylabel('PSNR')
    plt.title('Epoch Vs Peak Signal to Noise Ratio')
    plt.savefig("psnr.png")
    plt.show()

    plt.plot(epoch)
    plt.xlabel('epoch')
    plt.plot(losst, '-o')
    plt.ylabel('Train Loss')
    plt.title('Epoch Vs Training Loss ')
    plt.savefig("loss.png")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true',
                        help='train grayscale image denoising instead of RGB')
    parser.add_argument("--log_dir", type=str, default="./DLPF/Gray/50/",
                        help='path of log files')
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=50,
                        help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',
                        help="resume training from a previous checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Number of training steps to log psnr and perform orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5,
                        help="Number of training epochs to save state")
    parser.add_argument("--gnoiseIntL", nargs=2, type=int, default=[0, 75],
                        help="Noise training interval")
    parser.add_argument("--pnoiseIntL", nargs=2, type=int, default=[0, 20],
                        help="Noise training interval")
    parser.add_argument("--gval_noiseL", type=float, default=25,
                        help='noise level used on validation set')
    parser.add_argument("--pval_noiseL", type=float, default=10,
                        help='noise level used on validation set')

    argspar = parser.parse_args()
    argspar.gval_noiseL /= 255.
    argspar.gnoiseIntL[0] /= 255.
    argspar.gnoiseIntL[1] /= 255.

    print("\n### Training FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(argspar)
