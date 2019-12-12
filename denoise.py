import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deepglr.deepglr import *

try:
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:
    from skimage.measure import compare_ssim

import argparse


def main(args):
    sample = cv2.imread(args.input)
    shape = sample.shape
    if args.width:
        width = int(args.width)
    else:
        width = min(shape[0], shape[1])
    width = (width // 36) * 36
    if width < 36:
        print("Too small image, can't denoised")
        return 1
    if (
        args.width
        or (shape[0] != shape[1])
        or (shape[0] % 36) != 0
        or (shape[1] % 36) != 0
    ):
        sample = cv2.resize(sample, (width, width))
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample = sample.transpose((2, 0, 1))
    sample = torch.from_numpy(sample)

    cuda = True if torch.cuda.is_available() else False
    glr = DeepGLR(width=36, cuda=cuda)
    device = torch.device("cuda") if cuda else torch.device("cpu")
    glr.load_state_dict(torch.load(args.model, map_location=device))
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    psnrs = list()
    if args.ref:
        ref = cv2.imread(args.ref)
        if ref.shape[0] != width or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, width))

        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        tref = ref.copy()
        ref = ref.transpose((2, 0, 1))
        ref = torch.from_numpy(ref)

    tstart = time.time()
    T1 = sample
    if args.ref:
        T1r = ref
        print(T1r.shape, T1.shape)
    else:
        print(T1.shape)
    m = T1.shape[-1]
    dummy = np.zeros(shape=(3, T1.shape[-1], T1.shape[-2]))
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, 36, 36)
        .unfold(1, 36, 36)
    ).type(dtype)
    if args.ref:
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
        if args.ref:
            img1 = T2r[i, :, :, :, :].float()
            img2 = P
            psnrs.append(cv2.PSNR(img1.detach().numpy(), img2.detach().numpy()))

        print("\r{0}, {1}/{2}".format(P.shape, ii + 1, P.shape[0]), end=" ")
        for b, j in enumerate(range(0, m, s2)):
            dummy[:, (i * s2) : (i * s2 + s2), j : (j + s2)] = P[b].detach().numpy()
    print("\nPrediction time: ", time.time() - tstart)
    if args.ref:
        print("PSNR: ", np.mean(np.array(psnrs)))

    ds = np.array(dummy).copy()
    new_d = list()
    for d in ds:
        _d = (d - d.min()) * (1 / (d.max() - d.min()))
        new_d.append(_d)
    d = np.array(new_d).transpose(1, 2, 0)
    if args.output:
        opath = args.output
    else:
        filename = args.input.split("/")[-1]
        opath = "./{0}_{1}".format("denoised", filename)
    plt.imsave(opath, d)
    if args.ref:
        d = cv2.imread(opath)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        (score, diff) = compare_ssim(tref, d, full=True, multichannel=True)
        print("SSIM: ", score)
    print("Saved ", opath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-m", "--model", help="Path to the trained DeepGLR")
    parser.add_argument(
        "-o",
        "--output",
        help="Full output destination (with file name). Default is current working directory",
    )
    parser.add_argument("-r", "--ref", help="Reference image path (optional)")
    parser.add_argument(
        "-w", "--width", help="Resize image to a square image with given width"
    )

    args = parser.parse_args()

    main(args)
