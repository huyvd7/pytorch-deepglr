import time
import os
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
        width = 324
    if width <= 36:
        print("Too small image, can't denoised. Minimum width is 72")
        return 11

    if args.learning_rate:
        lr = float(args.learning_rate)
    else:
        lr = 2e-4
    if args.epoch:
        total_epoch = int(args.epoch)
    else:
        total_epoch = 200
    if args.destination:
        DST = args.destination
    else:
        DST = "./"

    if args.name:
        PATH = os.path.join(DST, args.name)
    else:
        PATH = os.path.join(DST, "DGLR.pkl")

    dataset = RENOIR_Dataset(
        img_dir=args.train_path,
        transform=transforms.Compose(
            [standardize(scale=None, w=width, normalize=True), ToTensor()]
        ),
    )
    patch_splitting(dataset=dataset, output_dst=DST, patch_size=36)

    dataset = RENOIR_Dataset(
        img_dir=os.path.join(DST, "patches"),
        transform=transforms.Compose([standardize(), ToTensor()]),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    glr = DeepGLR(width=36, cuda=cuda)
    if args.stack:
        print("Stacking single GLR", args.stack)
        glr.load(args.stack,args.stack,args.stack,args.stack)

    elif args.model:
        print("Continue training from: ", args.model)
        device = torch.device("cuda") if cuda else torch.device("cpu")
        try:
            glr.load_state_dict(torch.load(args.model, map_location=device))
        except Exception:
            print("Can't load model")
            return

    criterion = nn.MSELoss()
    momentum = 0.9
    gtv1_params = list(filter(lambda kv: 'gtv1' in kv[0] , glr.named_parameters()))
    gtv1_params = [i[1] for i in gtv1_params ]
    gtv2_params = list(filter(lambda kv: 'gtv2' in kv[0], glr.named_parameters()))
    gtv2_params = [i[1] for i in gtv2_params]

    #optimizer = optim.AdamW(glr.parameters(), lr=lr)
    optimizer = optim.SGD([
                {'params': gtv2_params, 'lr':lr},
                 {'params': gtv1_params , 'lr': lr*50}
             ], lr=lr, momentum=momentum)


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
            if epoch==0 and (i+1)%80==0:
                g = glr.glr1
                with torch.no_grad():
                    histW = g(inputs, debug=1)
                with torch.no_grad():
                    us = g.cnnu(inputs)
                    print("\tCNNU stats: ", us.max().data,  us.mean().data,us.min().data)
                g = glr.glr2
                with torch.no_grad():
                    histW = g(inputs, debug=1)
                g = glr.glr3
                with torch.no_grad():
                    histW = g(inputs, debug=1)

        print(
            time.ctime(),
            "[{0}] loss: {1:.3f}, time elapsed: {2}".format(
                epoch + 1, running_loss / (i + 1), time.time() - tstart
            ),
        )
            
        if (epoch + 1) % 1 == 0:
            print("save @ epoch ", epoch + 1)
            torch.save(glr.state_dict(), PATH)
            g = glr.glr1
            with torch.no_grad():
                histW = g(inputs, debug=1)
            with torch.no_grad():
                us = g.cnnu(inputs)
                print("\tCNNU stats: ", us.max().data,  us.mean().data,us.min().data)
            g = glr.glr2
            with torch.no_grad():
                histW = g(histW, debug=1)
            g = glr.glr3
            with torch.no_grad():
                histW = g(histW, debug=1)



    torch.save(glr.state_dict(), PATH)
    print("Total running time: {0:.3f}".format(time.time() - tstart))
    cleaning(DST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="Train set directory")
    parser.add_argument(
        "-m",
        "--model",
        help="Path to the trained DeepGLR. Will train from scratch if not specified",
    )
    parser.add_argument(
        "-n", "--name", help="Name of model. Default is DGLR.pkl",
    )
    parser.add_argument(
        "-d",
        "--destination",
        help="Output destination. Default is current working directory",
    )
    parser.add_argument(
        "-w",
        "--width",
        help="Resize image to a square image with given width before patch splitting. Default is 324. Minimum is 72",
    )
    parser.add_argument("-e", "--epoch", help="Total epochs")
    parser.add_argument(
        "-b", "--batch_size", help="Training batch size. Default is 100"
    )
    parser.add_argument(
        "-l", "--learning_rate", help="Training learning rate. Default is 2e-4"
    )
    parser.add_argument(
        "--stack", default=None
    )
    args = parser.parse_args()

    main(args)
