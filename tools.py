import torch


def state_to_corners(x, y, h, l, w):
    """returns B x NA x 4 x 2 tensor of corner coordinates
    """
    B, NA = x.shape
    xs = torch.stack((l, l, -l, -l), 2)/2.0
    ys = torch.stack((w, -w, -w, w), 2)/2.0
    hcos = h.cos().view(B, NA, 1)
    hsin = h.sin().view(B, NA, 1)

    xrot = xs * hcos - ys * hsin + x.view(B, NA, 1)
    yrot = xs * hsin + ys * hcos + y.view(B, NA, 1)

    return torch.stack((xrot, yrot), 3)


def state_to_corners_time(x, y, h, l, w):
    """
    x,y,h are B x NA x T
    l,w are B x NA x 1, or B x NA x T

    returns B x NA x T x 4 x 2 tensor of corner coordinates
    """
    B,NA,_ = l.shape
    _,_,T = x.shape
    xs = torch.stack((l, l, -l, -l), 3)/2.0
    ys = torch.stack((w, -w, -w, w), 3)/2.0
    hcos = h.cos().view(B, NA, T, 1)
    hsin = h.sin().view(B, NA, T, 1)

    xrot = xs * hcos - ys * hsin + x.view(B, NA, T, 1)
    yrot = xs * hsin + ys * hcos + y.view(B, NA, T, 1)

    return torch.stack((xrot, yrot), 4)


def calculate_car_collisions(x, y, h, l, w, k=None):
    """returns B x NA tensor of whether a car intersects with another or not
    k is an optional mask for which vehicles are real vs. just padding
    """
    corners = state_to_corners(x, y, h, l, w)
    collidex, collidey = state_corner_collisions(corners, x, y, h, l, w)
    no_crash = collidex & collidey & collidex.permute(0, 2, 1) & collidey.permute(0, 2, 1)
    if k is not None:
        no_crash = no_crash & k.unsqueeze(1) & k.unsqueeze(2)
    # ignore self-collisions
    no_crash = no_crash.sum(2) < 2
    return no_crash
