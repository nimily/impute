from .utils import soft_svt


def update(y, z_old, proj, scale, lambda_):
    level = scale * lambda_

    z_raw = y + scale * z_old - proj(z_old)
    z_new = soft_svt(z_raw, level)

    return z_new


def fit():
    pass


def fit_all():
    pass
