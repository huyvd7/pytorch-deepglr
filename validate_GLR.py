import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deepglr.deepglr import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:
    from skimage.measure import compare_ssim
import argparse


def main(args):
    if args.width:
        width = int(args.width) // 36 * 36
    else:
        width = 324
    if width < 36:
        print("Too small image, can't denoised")
        return 1
    testset = RENOIR_Dataset(
        img_dir=args.testdir,
        transform=transforms.Compose([standardize(w=width), ToTensor()]),  # 0.36
    )
    dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    cuda = True if torch.cuda.is_available() else False
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    glr = GLR(width=36, cuda=cuda)
    device = torch.device("cuda") if cuda else torch.device("cpu")
    glr.load_state_dict(torch.load(args.model, map_location=device))
    print("CUDA: {0}, device: {1}".format(cuda, device))
    psnrs = list()
    _psnrs = list()
    _ssims = list()
    psnrs_fullimage = list()
    tstart = time.time()
    for imgidx, sample in enumerate(dataloader):
        T1, T1r = sample["nimg"].squeeze(0).float(), sample["rimg"].squeeze(0).float()
        print(T1r.shape, T1.shape)
        m = T1.shape[-1]
        dummy = np.zeros(shape=(3, T1.shape[-1], T1.shape[-2]))
        T2 = (
            torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
            .unfold(0, 36, 36)
            .unfold(1, 36, 36)
        ).type(dtype)
        T2r = (
            torch.from_numpy(T1r.detach().numpy().transpose(1, 2, 0))
            .unfold(0, 36, 36)
            .unfold(1, 36, 36)
        )

        s2 = int(T2.shape[-1])

        for ii, i in enumerate(range(T2.shape[1])):
            P = glr.predict(T2[i, :, :, :, :].float())
            img1 = T2r[i, :, :, :, :].float()
            if cuda:
                P = P.cpu()

            img2 = P
            psnrs.append(cv2.PSNR(img1.detach().numpy(), img2.detach().numpy()))

            print("\r{0}, {1}/{2}".format(P.shape, ii + 1, P.shape[0]), end=" ")
            for b, j in enumerate(range(0, m, s2)):
                dummy[:, (i * s2) : (i * s2 + s2), j : (j + s2)] = P[b].detach().numpy()
        print("\nPrediction time: ", time.time() - tstart)
        print("PSNR: ", np.mean(np.array(psnrs)))
        _psnrs.append(np.mean(np.array(psnrs)))
        ds = np.array(dummy).copy()
        new_d = list()
        for d in ds:
            _d = (d - d.min()) * (1 / (d.max() - d.min()))
            new_d.append(_d)
        d = np.array(new_d).transpose(1, 2, 0)
        if args.output:
            opath = os.path.join(args.output, str(imgidx) + ".png")
            opathr = os.path.join(args.output, str(imgidx) + "_ref.png")
        else:
            opath = "./{0}{1}".format(imgidx, ".png")
            opathr = "./{0}{1}".format(imgidx, "_ref.png")
        print(d.min(), d.max(), end='-')
        d = np.minimum(np.maximum(d, 0), 1)
        print(d.min(), d.max())
        plt.imsave(opath, d)
        d = cv2.imread(opath)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        tref = sample["rimg"].squeeze(0)
        tref = tref.detach().numpy().transpose((1, 2, 0))
        plt.imsave(opathr, tref)
        _psnr_fullimage = cv2.PSNR(tref,d)
        psnrs_fullimage.append(_psnr_fullimage)
        (score, diff) = compare_ssim(tref, d, full=True, multichannel=True)
        _ssims.append(score)
        print("SSIM: ", score)
        print("PSNR (image-based): ", _psnr_fullimage)
        print("Saved ", opath)
    print("Mean PSNR (patch-based): {0:.3f}".format(np.mean(_psnrs)))
    print("Mean PSNR (image-based): {0:.3f}".format(np.mean(psnrs_fullimage)))
    print("Mean SSIM: {0:.3f}".format(np.mean(_ssims)))
    print("Total running time: {0:.3f}".format(time.time() - tstart))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testdir", help="Test set directory")
    parser.add_argument("-m", "--model", help="Path to the trained DeepGLR")
    parser.add_argument(
        "-o", "--output", help="Output directory. Default is current working directory",
    )

    parser.add_argument(
        "-w", "--width", help="Resize image to a square image with given width"
    )

    args = parser.parse_args()

    main(args)
