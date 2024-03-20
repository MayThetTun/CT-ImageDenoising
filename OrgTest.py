import argparse
import glob
import os
import os
import cv2
import matplotlib.pyplot
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from orgmodels import FFDNet
from utils import batch_psnr, batch_ssim, batch_fsim, batch_fsimgray, normalize, remove_dataparallel_wrapper, batch_vif
import lpips

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_ffdnet(**args):
    global gnoise
    in_ch = argspar.no_of_channel
    print("No of Channels", in_ch)
    print('Loading data info ...\n')
    path = r'data/NDCT/testing/*.png';
    files_source = glob.glob(path)
    files_source.sort()
    lpfunc = lpips.LPIPS(net='vgg').cuda()
    lpfunc = lpips.LPIPS(net='vgg').cuda()
    print("File source", len(files_source))
    psnr_test = 0
    ssim_test = 0
    fsim_test = 0
    lpips_test = 0
    vif_test = 0
    if in_ch == 3:
        model_fn = './LDCTLogs/OrgGray/80/best_model.pth'
        print("model used Path", model_fn)
    else:
        model_fn = './OrgMixedNoise/best_model.pth'
        print("models is ", model_fn)
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            model_fn)
    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # To know number of parameters work in CNN
    total_params = sum(
        param.numel() for param in net.parameters()
    )
    print("Parameters", total_params)
    if args['cuda']:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids)
        model.to(device)
        print("Device", model.to(device))
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)
    model.eval()
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    for f in files_source:
        if in_ch == 3:
            image = cv2.imread(f)
            imorig = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            imorig = np.expand_dims(image, 0)

        imorig = np.expand_dims(imorig, 0)
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape

        if sh_im[2] % 2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig,
                                     imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3] % 2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig,
                                     imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

        imorig = normalize(imorig)
        imorig = torch.FloatTensor(imorig)

        if args['add_noise']:
            gnoise = torch.FloatTensor(imorig.size()).normal_(std=args['gnoise_sigma'])
            peak = args['pnoise_sigma']
            impnoisy = torch.poisson(imorig * peak) / peak
            pnoise = impnoisy - imorig
            imnoisy = imorig + gnoise + pnoise
        else:
            imnoisy = imorig.clone()
        # Convert to GPU
        cnoise_sigma = args['gnoise_sigma'] + args['pnoise_sigma']
        with torch.no_grad():
            imorig, imnoisy = Variable(imorig.type(dtype)), \
                              Variable(imnoisy.type(dtype))
            nsigma = Variable(
                torch.FloatTensor([cnoise_sigma]).type(dtype))

        if in_ch == 1:
            outim = torch.clamp(
                imnoisy[:, :1, :, :].to('cuda') - model(imnoisy.to('cuda'), nsigma.to('cuda')),
                0., 1.)

        else:
            outim = torch.clamp(
                imnoisy[:, :3, :, :].to('cuda') - model(imnoisy.to('cuda'), nsigma.to('cuda')),
                0., 1.)

        # To handle width and height different Image size
        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]
            imnoisy = imnoisy[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
            imnoisy = imnoisy[:, :, :, :-1]

        # Evaluate the IQA Measurements(PSNR/SSIM/FSIM)

        psnr = batch_psnr(outim, imorig, 1.)
        ssims = batch_ssim(outim, imorig, 1.)
        if in_ch == 3:
            fsims = batch_fsim(outim, imorig, 1.)
        else:
            fsims = batch_fsimgray(outim, imorig, 1.)

        # lpips_value = lpfunc(outim, imorig).item()
        # vifs = batch_vif(outim, imorig, 1.)

        print("\n%s PSNR: %.3f" % (f, psnr))
        print("%s SSIM: %.3f" % (f, ssims))
        print("%s FSIM: %.3f" % (f, fsims))
        # print("%s LPIPS: %.3f" % (f, lpips_value))
        # print("%s VIF: %.3f" % (f, vifs))

        # To Display Noisy Images and Denoised Images
        # For Noisy Images
        if in_ch == 1:
            model_image = torch.squeeze(imnoisy[:, :1, :, :])
        else:
            model_image = torch.squeeze(outim[:, :3, :, :])
        # For Denoised Images
        model_image = torch.squeeze(model_image)
        model_image = transforms.ToPILImage()(model_image)
        fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
        ax0.imshow(model_image).set_cmap("gray")
        ax0.get_xaxis().set_ticks([])
        ax0.get_yaxis().set_ticks([])
        matplotlib.pyplot.box(False)
        matplotlib.pyplot.savefig("noisy_in_img.png", bbox_inches="tight")
        matplotlib.pyplot.show()
        matplotlib.pyplot.close(fig0)

        psnr_test += psnr
        ssim_test += ssims
        fsim_test += fsims
        # lpips_test += lpips_value
        # vif_test += vifs

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    fsim_test /= len(files_source)
    # lpips_test /= len(files_source)
    # vif_test /= len(files_source)

    print("\nPSNR : %.3f" % psnr_test)
    print("\nSSIM : %.3f" % ssim_test)
    print("\nFSIM : %.3f" % fsim_test)
    # print("\nLPIPS : %.3f" % lpips_test)
    # print("\nVIF : %.3f" % vif_test)

    # print("{:.3f}".format(psnr_test))
    # print("{:.4f}".format(ssim_test))
    # print("{:.4f}".format(fsim_test))
    # print("{:.4f}".format(lpips_test))
    # print("{:.3f}".format(vif_test))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise', type=str, default="true")
    parser.add_argument("--test_data", type=str, default="./data/gray/Set12",
                        help='path to input image')
    parser.add_argument("--gnoise_sigma", type=float, default=25, help="Gaussian Noise Interval")
    parser.add_argument("--pnoise_sigma", type=float, default=10, help="Poisson Noise Interval")
    parser.add_argument("--no_gpu", action='store_true',
                        help="run model on CPU")
    parser.add_argument("--no_of_channel", type=int, default=1, help="color for 3 and grayscale for 1")
    argspar = parser.parse_args()
    argspar.gnoise_sigma /= 255.
    argspar.add_noise = (argspar.add_noise.lower() == 'true')
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()
    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    test_ffdnet(**vars(argspar))
