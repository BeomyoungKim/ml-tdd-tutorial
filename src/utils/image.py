import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.constants import CLASSES


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


def get_classes_and_probs(outputs):
    _, preds_tensor = torch.max(outputs, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]


def plot_classes_preds(outputs, images, labels):
    preds, probs = get_classes_and_probs(outputs)
    fig = plt.figure(figsize=(6.4, 4.8))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                CLASSES[preds[idx]],
                probs[idx] * 100.0,
                CLASSES[labels[idx]]
            ),
            color=("green" if preds[idx]==labels[idx].item() else "red")
        )
    return fig
