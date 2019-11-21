import torch
import numpy as np
import scipy.sparse as ss


def get_w(ij, F):
    fi, fj = F[:, :, ij[0]], F[:, :, ij[1]]
    d = dist(fi, fj)
    return w(d)

def dist(fi, fj):
    return torch.sum((fi-fj)**2, axis=1)


def w(d, epsilon=1):
    return torch.exp(-d/(2*epsilon**2))


def connected_adjacency(image, connect=8, patch_size=(1, 1)):
    r, c = image.shape[:2]
    r = int(r / patch_size[0])
    c = int(c / patch_size[1])

    if connect == '4':
        # constructed from 2 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
        d2 = np.ones(c*(r-1))
        upper_diags = ss.diags([d1, d2], [1, c])
        return upper_diags + upper_diags.T

    elif connect == '8':
        # constructed from 4 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
        d2 = np.append([0], d1[:c*(r-1)])
        d3 = np.ones(c*(r-1))
        d4 = d2[1:-1]
        upper_diags = ss.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
        return upper_diags + upper_diags.T

def adjacency_construction(width, F, ntype='8'):
    if type(F)!=torch.Tensor:
        F = torch.from_numpy(F)
    with torch.no_grad():
    # 8 connected pixel structure
        pixel_indices = [i for i in range(width*width)]
        pixel_indices = np.reshape(pixel_indices, (width, width))
        A = connected_adjacency(pixel_indices, ntype)
        A_pair = np.asarray(np.where(A.toarray() == 1)).T
        A_pair = torch.from_numpy(A_pair)
        def lambda_func(x): return get_w(x, F)
        W = torch.tensor(list(map(lambda_func, A_pair)))
        A = A.toarray()
        A = torch.from_numpy(A)
        
        for idx, p in enumerate(A_pair):
            i = p[0]
            j = p[1]
            A[i][j] = W[idx]

    return A.numpy()


def laplacian_construction(width, F, ntype='8'):
    if type(F)!=torch.Tensor:
        F = torch.from_numpy(F)
    with torch.no_grad():
    # 8 connected pixel structure
        pixel_indices = [i for i in range(width*width)]
        pixel_indices = np.reshape(pixel_indices, (width, width))
        A = connected_adjacency(pixel_indices, ntype)
        A_pair = np.asarray(np.where(A.toarray() == 1)).T
        A_pair = torch.from_numpy(A_pair)
        def lambda_func(x): return get_w(x, F)
        W = list(map(lambda_func, A_pair))
        A = torch.zeros(F.shape[0], width**2, width**2)
        for idx, p in enumerate(A_pair):
            i = p[0]
            j = p[1]
            A[:, i, j] = W[idx]
        D = torch.diag_embed(torch.sum(A, axis=1), offset=0, dim1=-2, dim2=-1)
        L = D - A
    return L


def qpsolve(L, u, y, I, wt=26):
    xhat = torch.inverse(I + u[:, None]*L)
    xhat = torch.bmm(xhat, y) 
    return xhat