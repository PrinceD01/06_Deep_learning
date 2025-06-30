import torch
from functools import partial

def pairwise_distance(x, y):
    """
    Cette fonction calcule les distances euclidiennes par paires entre deux matrices x et y.
    Les deux matrices doivent avoir deux dimensions et leur deuxième dimension doit avoir le même nombre de caractéristiques.
    Elle calcule la distance euclidienne au carré entre chaque paire de vecteurs le long de la première dimension du tenseur de sortie et renvoie le tenseur de forme résultant (batch_size, batch_size).
    """
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):
    """
    La fonction gaussian_kernel_matrix calcule la matrice de noyau gaussienne entre deux ensembles de points x et y, étant donné un vecteur de largeurs de noyau (sigmas).
    Il calcule d'abord les distances euclidiennes par paires entre chaque point dans x et y en utilisant la fonction pairwise_distance.
    Il applique ensuite la fonction de noyau gaussien aux distances par paires à l'aide des largeurs de noyau et renvoie la matrice de noyau résultante.

    La fonction noyau gaussienne est définie comme suit : K(x,y) = exp(-||xy||^2 / 2*sigma^2) où sigma est la largeur du noyau et ||xy||^2 est le carré Distance euclidienne entre x et y.
    """
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    """
    La fonction maximum_mean_discrepancy calcule l'écart moyen maximal (MMD) entre deux ensembles d'échantillons x et y.
    Le MMD est une métrique de distance utilisée pour mesurer la différence entre deux distributions de probabilité.
    Le MMD est calculé à l'aide de la fonction de matrice du noyau gaussien par défaut et il représente la différence entre la somme des matrices du noyau de x et y, et deux fois la matrice du noyau de x et y.
    """
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mmd_loss(source_features, target_features, use_gpu=True):
    """
    La fonction mmd_loss prend deux arguments : source_features et target_features. Ces deux arguments sont des tenseurs représentant les cartes de caractéristiques des domaines source et cible, respectivement.
    La fonction calcule l'écart moyen maximal (MMD) entre le source et le cible qu'on utilise pour quantifier la différence entre les distributions de caractéristiques des domaines source et cible.
    La fonction calcule MMD à l'aide d'une fonction de noyau gaussien avec une plage de différentes valeurs du paramètre de bande passante (sigmas).
    La fonction noyau gaussien est définie dans la fonction gaussian_kernel_matrix .
    La fonction renvoie la perte MMD calculée sous forme de tenseur PyTorch.
    """
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if use_gpu:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value