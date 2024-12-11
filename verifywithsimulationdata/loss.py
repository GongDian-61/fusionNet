import torch

def GradNorm(disp):
    """
    size of disp = ??
    """

    dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
    dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    dxx = torch.abs(dx[:, :, :, 1:] - dx[:, :, :, :-1])
    dyy = torch.abs(dy[:, :, 1:, :] - dy[:, :, :-1, :])
    # mean of which dim?
    d = (torch.mean(dxx) + torch.mean(dyy)) / 2
    return d

