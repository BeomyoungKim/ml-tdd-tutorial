import torch
import numpy as np
import matplotlib.pyplot as plt


def matplotlib_imshow(img: torch.Tensor, one_channel: bool = False):
    if one_channel:
        img = img.mean(dim=0)
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        # HWC
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
