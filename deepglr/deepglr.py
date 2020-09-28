import scipy.sparse as ss
import torch
import numpy as np
import os
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


class cnnf(nn.Module):
    """
    CNN F of GLR
    """

    def __init__(self):
        super(cnnf, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(0.05)
        )
        self.layer2a = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        self.layer3a = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # self.maxpool
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # DECONVO

        self.deconvo1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
        )

        # CONCAT with output of layer2
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(0.05)
        )
        # DECONVO
        self.deconvo2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
        )

        # CONCAT with output of layer1
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        outl1 = self.layer1(x)
        outl2 = self.layer2a(outl1)
        outl2 = self.maxpool(outl2)
        outl2 = self.layer2(outl2)
        outl3 = self.layer3a(outl2)
        outl3 = self.maxpool(outl3)
        outl3 = self.layer3(outl3)
        outl3 = self.deconvo1(outl3)
        outl3 = torch.cat((outl3, outl2), dim=1)
        outl4 = self.layer4(outl3)
        outl4 = self.deconvo2(outl4)
        outl4 = torch.cat((outl4, outl1), dim=1)
        del outl1, outl2, outl3
        out = self.layer5(outl4)
        return out


class cnnf2(nn.Module):
    def __init__(self):
        super(cnnf, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(0.05),
            nn.Conv2d(32, 6, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # identity = x
        out = self.layer(x)
        # out = identity + out
        return out




class cnny(nn.Module):
    """
    CNN Y of GLR
    """

    def __init__(self):
        super(cnny, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out = identity + out
        del identity
        return out


class cnnu(nn.Module):
    """
    CNNU of GLR
    """
    def __init__(self):
        super(cnnu, self).__init__()
        self.layer = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 32, 1 * 1 * 32),
            nn.Linear(1 * 1 * 32, 1),
            nn.ReLU()
            # nn.LeakyReLU(0.05),
        )


    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


class RENOIR_Dataset(Dataset):
    """
    Dataset loader
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.npath = os.path.join(img_dir, "noisy")
        self.rpath = os.path.join(img_dir, "ref")
        self.nimg_name = sorted(os.listdir(self.npath))
        self.rimg_name = sorted(os.listdir(self.rpath))
        self.nimg_name = [
            i
            for i in self.nimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp"]
        ]
        self.rimg_name = [
            i
            for i in self.rimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp"]
        ]
        self.transform = transform

    def __len__(self):
        return len(self.nimg_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        nimg_name = os.path.join(self.npath, self.nimg_name[idx])
        nimg = cv2.imread(nimg_name)
        rimg_name = os.path.join(self.rpath, self.rimg_name[idx])
        rimg = cv2.imread(rimg_name)

        sample = {"nimg": nimg, "rimg": rimg}

        if self.transform:
            sample = self.transform(sample)

        return sample


class standardize(object):
    """Convert opencv BGR to RGB order. Scale the image with a ratio"""

    def __init__(self, scale=None, w=None, normalize=None):
        """
        Args:
        scale (float): resize height and width of samples to scale*width and scale*height
        width (float): resize height and width of samples to width x width. Only works if "scale" is not specified
        """
        self.scale = scale
        self.w = w
        self.normalize = normalize

    def __call__(self, sample):
        nimg, rimg = sample["nimg"], sample["rimg"]
        if self.scale:
            nimg = cv2.resize(nimg, (0, 0), fx=self.scale, fy=self.scale)
            rimg = cv2.resize(rimg, (0, 0), fx=self.scale, fy=self.scale)
        else:
            if self.w:
                nimg = cv2.resize(nimg, (self.w, self.w))
                rimg = cv2.resize(rimg, (self.w, self.w))
        if self.normalize:
            nimg = cv2.resize(nimg, (0, 0), fx=1, fy=1)
            rimg = cv2.resize(rimg, (0, 0), fx=1, fy=1)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        if self.normalize:
            nimg = nimg / 255
            rimg = rimg / 255
        return {"nimg": nimg, "rimg": rimg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Swap color axis from H x W x C (numpy) to C x H x W (torch)
        """
        nimg, rimg = sample["nimg"], sample["rimg"]
        nimg = nimg.transpose((2, 0, 1))
        rimg = rimg.transpose((2, 0, 1))
        return {"nimg": torch.from_numpy(nimg), "rimg": torch.from_numpy(rimg)}


def connected_adjacency(image, connect=8, patch_size=(1, 1)):
    """
    Construct 8-connected pixels base graph (0 for not connected, 1 for connected)
    """
    r, c = image.shape[:2]
    r = int(r / patch_size[0])
    c = int(c / patch_size[1])

    if connect == "4":
        # constructed from 2 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.ones(c * (r - 1))
        upper_diags = ss.diags([d1, d2], [1, c])
        return upper_diags + upper_diags.T

    elif connect == "8":
        # constructed from 4 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.append([0], d1[: c * (r - 1)])
        d3 = np.ones(c * (r - 1))
        d4 = d2[1:-1]
        upper_diags = ss.diags([d1, d2, d3, d4], [1, c - 1, c, c + 1])
        return upper_diags + upper_diags.T


def get_w(ij, F):
    """
    Compute weights for node i and node j using exemplars F
    """
    fi, fj = F[:, :, ij[0]], F[:, :, ij[1]]
    d = dist(fi, fj)
    return w(d).type(dtype)


def w(d, epsilon=1):
    """
    Compute (3)
    """
    return torch.exp(-d / (2 * epsilon ** 2))


def dist(fi, fj):
    """
    Compute the distance using equation (4)
    """
    return torch.sum((fi - fj) ** 2, axis=1).type(dtype)


def laplacian_construction(width, F, ntype="8", debug=False):
    """
    Construct Laplacian matrix
    """
    if type(F) != torch.Tensor:
        F = torch.from_numpy(F)
    with torch.no_grad():
        pixel_indices = [i for i in range(width * width)]
        pixel_indices = np.reshape(pixel_indices, (width, width))
        A = connected_adjacency(pixel_indices, ntype)
        A_pair = np.asarray(np.where(A.toarray() == 1)).T

        def lambda_func(x):
            return get_w(x, F)



        W = list(map(lambda_func, A_pair))
        #Fs = (
        #    self.opt.H.matmul(E.view(E.shape[0], E.shape[1], self.opt.width ** 2, 1))
        #    ** 2  
        #)
        #w = torch.exp(-(Fs.sum(axis=1)) / (2 * (1 ** 2)))

        A = torch.zeros(F.shape[0], width ** 2, width ** 2).type(dtype)
        for idx, p in enumerate(A_pair):
            i = p[0]
            j = p[1]
            A[:, i, j] = W[idx]
        if debug:
            print("\t\x1b[31mWEIGHT SUM (1 sample)\x1b[0m", A[0, :, :].sum().item())

        D = torch.diag_embed(torch.sum(A, axis=1), offset=0, dim1=-2, dim2=-1).type(
            dtype
        )
        L = D - A
    return L.type(dtype)


def weights_init_normal(m):
    """
    Initialize weights of convolutional layers
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class OPT:
    def __init__(
        self,
        batch_size=100,
        width=36,
        connectivity="8",
        admm_iter=1,
        prox_iter=1,
        delta=1,
        channels=3,
        eta=0.1,
        u=1,
        u_max=100,
        u_min=10,
        lr=1e-4,
        momentum=0.99,
        ver=None,
        train="gauss_batch",
        cuda=False,
    ):
        self.batch_size = batch_size
        self.width = width
        self.edges = 0
        self.nodes = width ** 2
        self.I = None
        self.pairs = None
        self.H = None
        self.connectivity = connectivity
        self.admm_iter = admm_iter
        self.prox_iter = prox_iter
        self.channels = channels
        self.eta = eta
        self.u = u
        self.lr = lr
        self.delta = delta
        self.momentum = momentum
        self.u_max = u_max
        self.u_min = u_min
        self.ver = ver
        self.D = None
        self.train = train
        self.pg_zero = None
        self.cuda= cuda
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor


    def _print(self):
        print(
            "batch_size =",
            self.batch_size,
            ", width =",
            self.width,
            ", admm_iter =",
            self.admm_iter,
            ", prox_iter =",
            self.prox_iter,
            ", delta =",
            self.delta,
            ", channels =",
            self.channels,
            ", eta =",
            self.eta,
            ", u_min =",
            self.u_min,
            ", u_max =",
            self.u_max,
            ", lr =",
            self.lr,
            ", momentum =",
            self.momentum,
        )

def supporting_matrix(opt):
    dtype = opt.dtype
    cuda = opt.cuda
    width = opt.width

    pixel_indices = [i for i in range(width * width)]
    pixel_indices = np.reshape(pixel_indices, (width, width))
    A = connected_adjacency(pixel_indices, connect=opt.connectivity)
    A_pair = np.asarray(np.where(A.toarray() == 1)).T
    A_pair = np.unique(np.sort(A_pair, axis=1), axis=0)

    opt.edges = A_pair.shape[0]
    H_dim0 = opt.edges
    H_dim1 = width ** 2
    # unique_A_pair = np.unique(np.sort(A_pair, axis=1), axis=0)

    I = torch.eye(width ** 2, width ** 2).type(dtype)
    lagrange = torch.zeros(opt.edges, 1).type(dtype)
    A = torch.zeros(width ** 2, width ** 2).type(dtype)
    H = torch.zeros(H_dim0, H_dim1).type(dtype)
    for e, p in enumerate(A_pair):
        H[e, p[0]] = 1
        H[e, p[1]] = -1
        A[p[0], p[1]] = 1
        # A[p[1], p[0]] = 1

    opt.I = I  # .type(dtype).requires_grad_(True)
    opt.pairs = A_pair
    opt.H = H  # .type(dtype).requires_grad_(True)
    opt.connectivity_full = A.requires_grad_(True)
    opt.connectivity_idx = torch.where(A > 0)

    for e, p in enumerate(A_pair):
        A[p[1], p[0]] = 1
    opt.lagrange = lagrange  # .requires_grad_(True)
    opt.D = torch.inverse(2 * opt.I + opt.delta * (opt.H.T.mm(H))).type(dtype).detach()
    opt.pg_zero = torch.zeros(opt.edges, 1).type(dtype)
    print("OPT created on cuda:", cuda, dtype)


class GLR(nn.Module):
    """
    GLR network
    """

    def __init__(self, width=36, cuda=False, opt=None):
        super(GLR, self).__init__()
        self.cnnf = cnnf()
        self.cnny = cnny()
        self.cnnu = cnnu()
        self.wt = width
        self.identity_matrix = torch.eye(self.wt ** 2, self.wt ** 2).type(dtype)
        self.umax = (250 - 1) / (2 * 8)  # 15.5625
        if cuda:
            self.cnnf.cuda()
            self.cnny.cuda()
            self.cnnu.cuda()
            self.identity_matrix.cuda()
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.cnnf.apply(weights_init_normal)
        self.cnny.apply(weights_init_normal)
        self.cnnu.apply(weights_init_normal)
        self.opt = opt
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.support_L = torch.ones(self.opt.width**2, 1).type(self.dtype)
        supporting_matrix(self.opt)
        self.base_W = torch.zeros(self.opt.batch_size, self.opt.channels, self.opt.width ** 2, self.opt.width ** 2).type(self.dtype)


    def forward(self, xf, debug=False):
        E = self.cnnf.forward(xf).squeeze(0)
        Y = self.cnny.forward(xf).squeeze(0)
        u = self.cnnu.forward(xf)
            #u[u > 15.5625] = 15.5625
        u = torch.clamp(u, 0.001, 15.5625)
        u = u.unsqueeze(1).unsqueeze(1)

        img_dim = self.wt

        L = laplacian_construction(
            width=img_dim, F=E.view(E.shape[0], E.shape[1], img_dim ** 2), debug=debug
        )
        Fs = (
            self.opt.H.matmul(E.view(E.shape[0], E.shape[1], self.opt.width ** 2, 1))
            ** 2
        )
        W = self.base_W.clone()

        w = torch.exp(-(Fs.sum(axis=1)) / (2 * (1 ** 2)))
        w = w.unsqueeze(1).repeat(1, self.opt.channels, 1, 1)
        W[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = w.view(
            xf.shape[0], 3, -1
        )
        W[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = w.view(
            xf.shape[0], 3, -1
        )
        L1 = W @ self.support_L
        L = torch.diag_embed(L1.squeeze(-1)) - W
        

        out = qpsolve(
            L=L, u=u, y=Y.view(Y.shape[0], img_dim ** 2, 3), Im=self.identity_matrix
        )
        return out.view(xf.shape[0], 3, img_dim, img_dim)

    def predict(self, xf):
        E = self.cnnf.forward(xf).squeeze(0)
        Y = self.cnny.forward(xf).squeeze(0)
        u = self.cnnu.forward(xf)
        u[u > 15.5625] = 15.5625
        img_dim = self.wt
        identity_matrix = torch.eye(img_dim ** 2, img_dim ** 2).type(self.dtype)
        if xf.shape[0] == 1:
            E = E.unsqueeze(0)
            Y = Y.unsqueeze(0)
        E = E.view(E.shape[0], E.shape[1], img_dim ** 2)
        #Y = Y.view(Y.shape[0], img_dim ** 2, 3)

        L = laplacian_construction(width=img_dim, F=E)

        out = qpsolve(L=L, u=u, y=Y.contiguous().view(Y.shape[0], img_dim ** 2, 3), Im=identity_matrix)
        return out.view(xf.shape[0], 3, img_dim, img_dim)


class DeepGLR(nn.Module):
    """
    Stack 4 GLRs
    """

    def __init__(self, width=36, cuda=False):
        super(DeepGLR, self).__init__()
        self.glr1 = GLR(cuda=cuda)
        self.glr2 = GLR(cuda=cuda)
        self.glr3 = GLR(cuda=cuda)
        self.glr4 = GLR(cuda=cuda)
        self.cuda = cuda

        if self.cuda:
            self.glr1.cuda()
            self.glr2.cuda()
            self.glr3.cuda()
            self.glr4.cuda()

    def load(self, PATH1, PATH2, PATH3, PATH4):
        if self.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.glr1.load_state_dict(torch.load(PATH1, map_location=device))
        self.glr2.load_state_dict(torch.load(PATH2, map_location=device))
        self.glr3.load_state_dict(torch.load(PATH3, map_location=device))
        self.glr4.load_state_dict(torch.load(PATH4, map_location=device))

    def predict(self, sample):
        if self.cuda:
            sample.cuda()
        P = self.glr1.predict(sample)
        P = self.glr2.predict(P)
        P = self.glr3.predict(P)
        P = self.glr4.predict(P)
        return P

    def forward(self, sample):
        P = self.glr1.forward(sample)
        P = self.glr2.forward(P)
        P = self.glr3.forward(P)
        P = self.glr4.forward(P)
        return P


def qpsolve(L, u, y, Im):
    """
    Solve equation (2) using (6)
    """
    xhat = torch.inverse(Im + u[:, None] * L)
    return xhat @ y


def patch_splitting(dataset, output_dst, patch_size=36):
    """Split each image in the dataset to patch size with size patch_size x patch_size"""
    import shutil

    output_dst_temp = os.path.join(output_dst, "patches")
    output_dst_noisy = os.path.join(output_dst_temp, "noisy")
    output_dst_ref = os.path.join(output_dst_temp, "ref")
    try:
        if not os.path.exists(output_dst_temp):
            os.makedirs(output_dst_temp)
        else:
            shutil.rmtree(output_dst_temp)  # Removes all the subdirectories!
            os.makedirs(output_dst_temp)

        if not os.path.exists(output_dst_noisy):
            os.makedirs(output_dst_noisy)
        else:
            shutil.rmtree(output_dst_noisy)  # Removes all the subdirectories!
            os.makedirs(output_dst_noisy)

        if not os.path.exists(output_dst_ref):
            os.makedirs(output_dst_ref)
        else:
            shutil.rmtree(output_dst_ref)  # Removes all the subdirectories!
            os.makedirs(output_dst_ref)

    except Exception:
        print(
            "Cannot create temporary directories for saving image patches at ",
            output_dst,
        )

    dataloader = DataLoader(dataset, batch_size=1)
    total = 0
    for i_batch, s in enumerate(dataloader):
        nnn = dataset.nimg_name[i_batch]
        rn = dataset.rimg_name[i_batch]
        T1 = (
            s["nimg"]
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .reshape(1, 3, -1, patch_size, patch_size)
            .squeeze()
        )
        T2 = (
            s["rimg"]
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .reshape(1, 3, -1, patch_size, patch_size)
            .squeeze()
        )
        for i in range(T1.shape[1]):
            save_image(
                T1[:, i, :, :], os.path.join(output_dst_noisy, "{0}_{1}".format(i, nnn))
            )
            total += 1
        for i in range(T2.shape[1]):
            save_image(
                T2[:, i, :, :], os.path.join(output_dst_ref, "{0}_{1}".format(i, rn))
            )
    print("Total image patches: ", total)


def cleaning(output_dst):
    """Clean the directory after running"""
    import shutil

    output_dst_temp = os.path.join(output_dst, "patches")
    try:
        shutil.rmtree(output_dst_temp)  # Removes all the subdirectories!
    except Exception:
        print("Cannot clean the temporary image patches")
