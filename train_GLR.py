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
    supporting_matrix(opt)
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
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    glr = GLR(width=36, cuda=cuda, opt=opt)

    if args.model:
        print("Continue training from: ", args.model)
        device = torch.device("cuda") if cuda else torch.device("cpu")
        try:
            glr.load_state_dict(torch.load(args.model, map_location=device))
        except Exception:
            print("Can't load model")
            return

    criterion = nn.MSELoss()
    momentum = 0.9
    cnnf_params = list(filter(lambda kv: 'cnnf' in kv[0] , glr.named_parameters()))
    cnnf_params = [i[1] for i in cnnf_params]
    cnny_params = list(filter(lambda kv: 'cnny' in kv[0], glr.named_parameters()))
    cnny_params = [i[1] for i in cnny_params]
    cnnu_params = list(filter(lambda kv: 'cnnu' in kv[0], glr.named_parameters()))
    cnnu_params = [i[1] for i in cnnu_params]
    print(len(cnnf_params), len(cnny_params), len(cnnu_params))
    #optimizer = optim.AdamW(glr.parameters(), lr=lr)
    optimizer = optim.SGD([
                {'params': cnny_params, 'lr':lr/5},
                {'params': cnnu_params, 'lr':lr*10},
                 {'params': cnnf_params , 'lr': lr*50}
             ], lr=lr, momentum=momentum)



    tstart = time.time()
    ld=len(dataset)
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
            torch.nn.utils.clip_grad_norm_(cnnf_params, 1e1)
            torch.nn.utils.clip_grad_norm_(cnny_params, 1e0)
            torch.nn.utils.clip_grad_norm_(cnnu_params, 1e0)

            optimizer.step()

            running_loss += loss.item()
            if epoch==0 and (i+1)%80==0:

                g = glr

                with torch.no_grad():
                    histW = g(inputs, debug=1)
                print("\tCNNU grads: ", g.cnnu.layer[0].weight.grad.mean().item())
                print("\tCNNF stats: ", g.cnnf.layer1[0].weight.grad.mean().item())

                with torch.no_grad():
                    us = g.cnnu(inputs)
                    print("\tCNNU stats: ", us.max().item(),  us.mean().item(),us.min().item())



        print(
            time.ctime(),
            "[{0}] loss: {1:.3f}, time elapsed: {2}".format(
                epoch + 1, running_loss / (ld*(i + 1)), time.time() - tstart
            ),
        )
        if (epoch + 1) % 1 == 0:
            print("save @ epoch ", epoch + 1)
            torch.save(glr.state_dict(), PATH+'_{0}'.format(epoch))
            g = glr
            with torch.no_grad():
                histW = g(inputs, debug=1)
            print("\tCNNU grads: ", g.cnnu.layer[0].weight.grad.mean().item())
            print("\tCNNF stats: ", g.cnnf.layer1[0].weight.grad.mean().item())
            with torch.no_grad():
                us = g.cnnu(inputs)
                print("\tCNNU stats: ", us.max().item(),  us.mean().item(),us.min().item())





    torch.save(glr.state_dict(), PATH+'_{0}'.format(epoch))
    print("Total running time: {0:.3f}".format(time.time() - tstart))
    cleaning(DST)

opt = OPT(batch_size = 50, admm_iter=4, prox_iter=3, delta=.1, channels=3, eta=.05, u=50, lr=8e-6, momentum=0.9, u_max=65, u_min=50, cuda=True if torch.cuda.is_available() else False)

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
        help="Resize image to a square image with given width before patch splitting. Default is 324. Minimum is 72", type=int
    )
    parser.add_argument("-e", "--epoch", help="Total epochs")
    parser.add_argument(
        "-b", "--batch_size", help="Training batch size. Default is 100",type=int
    )
    parser.add_argument(
        "-l", "--learning_rate", help="Training learning rate. Default is 2e-4"
    )
    args = parser.parse_args()
    opt.batch_size = args.batch_size
    main(args)
