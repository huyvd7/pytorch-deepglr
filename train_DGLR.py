import time
import torch
import torch.nn as nn
from deepglr.deepglr import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

import argparse


def main(args):
    cuda = True if torch.cuda.is_available() else False
    print("CUDA: ", cuda)
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 100
    if args.width:
        width = int(args.width) // 36 * 36
    else:
        width = 36
    if width < 36:
        print("Too small image, can't denoised")
        return 1

    dataset = RENOIR_Dataset(
        img_dir=args.train_path,
        transform=transforms.Compose([standardize(scale=None, w=width), ToTensor()]),
    )

    if args.learning_rate:
        lr = float(args.learning_rate)
    else:
        lr = 2e-4
    if args.epoch:
        total_epoch = int(args.epoch)
    else:
        total_epoch = 200
    if args.output:
        PATH = args.output
    else:
        PATH = "./dglr.pkl"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    glr = DeepGLR(width=width, cuda=cuda)

    if args.model:
        print("Continue training from: ", args.model)
        device = torch.device("cuda") if cuda else torch.device("cpu")
        try:
            glr.load_state_dict(torch.load(args.model, map_location=device))
        except Exception:
            print("Can't load model")
            return

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(glr.parameters(), lr=lr)

    tstart = time.time()
    for epoch in range(total_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):  # start index at 0
            # get the inputs; data is a list of [inputs, labels]
            labels = data["rimg"].float().type(dtype)
            inputs = data["nimg"].float().type(dtype)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = glr(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(
            time.ctime(),
            "[{0}] loss: {1:.3f}, time elapsed: {2}".format(
                epoch + 1, running_loss / (i + 1), time.time() - tstart
            ),
        )
        if (epoch + 1) % 10 == 0:
            print("save @ epoch ", epoch + 1)
            torch.save(glr.state_dict(), PATH)

    print("Total running time: {0:.3f}".format(time.time() - tstart))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="Train set directory")
    parser.add_argument(
        "-m",
        "--model",
        help="Path to the trained DeepGLR. Will train from scratch if not specified",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output full path (with filename). Default is current working directory",
    )
    parser.add_argument(
        "-w",
        "--width",
        help="Resize image to a square image with given width. Default is 36",
    )
    parser.add_argument("-e", "--epoch", help="Total epochs")
    parser.add_argument(
        "-b", "--batch_size", help="Training batch size. Default is 100"
    )
    parser.add_argument(
        "-l", "--learning_rate", help="Training learning rate. Default is 2e-4"
    )
    args = parser.parse_args()

    main(args)
