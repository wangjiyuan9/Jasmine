import torch
import torch.nn as nn
import torch.nn.functional as F


# Scale and Shift Invariant Loss
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask):
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        prediction, target = prediction.squeeze(1), target.squeeze(1)
        # add
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target = target.float()

            scale, shift = compute_scale_and_shift_masked(prediction, target, mask)
            scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        return loss


def compute_scale_and_shift_masked(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0  # 1e-3
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1


class MedianLoss(nn.Module):
    def __init__(self, opt, device):
        super(MedianLoss, self).__init__()
        self.name = "median_loss"
        self.opt = opt
        crop = torch.tensor([
            int(0.40810811 * self.opt.height),
            int(0.99189189 * self.opt.height),
            int(0.03594771 * self.opt.width),
            int(0.96405229 * self.opt.width)
        ], dtype=torch.long, device=device)
        crop_mask = torch.zeros((self.opt.height, self.opt.width), device=device, dtype=torch.bool)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
        self.crop_mask = crop_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, prediction, target, loss_fn=None, depth_mask=None):
        batch_size = prediction.size(0)
        target_flat = target.view(batch_size, -1)
        mask_flat = self.crop_mask.expand(batch_size, -1, -1, -1).view(batch_size, -1)
        max_values = torch.quantile(target_flat[mask_flat].view(batch_size, -1), 0.9995, dim=1)

        target_clamped = torch.min(target, max_values.view(-1, 1, 1, 1))
        scale, shift = compute_scale_and_shift_masked(prediction.squeeze(), target_clamped.squeeze(), self.crop_mask.expand(batch_size, -1, -1, -1).squeeze())
        pred_scaled = (scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1))

        weight_mask = (self.crop_mask.expand(batch_size, -1, -1, -1) + 0.1) * depth_mask
        # target_clamped = torch.min(target_flat, max_values.unsqueeze(1).expand_as(target_flat))
        # pred, tgt = prediction_flat[mask_flat], target_clamped[mask_flat]
        #
        # median_target = tgt.view(batch_size, -1).median(dim=1).values
        # median_prediction = pred.view(batch_size, -1).median(dim=1).values
        #
        # ratio = median_target / (median_prediction + 1e-8)
        # pred_scaled = pred.view(batch_size, -1) * ratio.unsqueeze(1)
        if loss_fn is not None:
            loss = loss_fn(pred_scaled, target_clamped) * weight_mask
        else:
            loss = nn.functional.l1_loss(pred_scaled, target_clamped, reduction='none') * weight_mask
        return loss.mean()


class median_loss(nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.name = "median_loss"
        self.opt = opt
        crop = torch.tensor([0.40810811 * self.opt.height, 0.99189189 * self.opt.height, 0.03594771 * self.opt.width, 0.96405229 * self.opt.width], dtype=torch.long, device=device)
        crop_mask = torch.zeros((self.opt.height, self.opt.width), device=device, dtype=torch.bool)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
        self.crop_mask = crop_mask

    def forward(self, prediction, target):
        bs = prediction.size(0)
        mask = self.crop_mask
        prediction, target = prediction.squeeze(1), target.squeeze(1)
        loss = 0
        for b in range(bs):
            prediction_m, target_m = prediction[b][mask], target[b][mask]
            max_value = torch.quantile(target_m, 0.999)
            target_m = torch.clamp(target_m, 0, max_value)
            target[b] = torch.clamp(target[b], 0, max_value)
            ratio = torch.median(target_m) / torch.median(prediction_m)
            prediction[b] = prediction[b] * ratio
            loss += nn.functional.l1_loss(prediction[b], target[b])
        loss = loss / bs
        return loss


# Angluar Loss
class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()
        self.name = "Angular"

    def forward(self, prediction, target, mask=None):
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target = target.float()
            mask = mask[:, 0, :, :]
            dot_product = torch.sum(prediction * target, dim=1)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            angle = torch.acos(dot_product)
            if mask is not None:
                angle = angle[mask]
            loss = angle.mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None, d_map=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = torch.abs(target - pred)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()
        part1 = -F.threshold(-diff, -delta, 0.)  # 筛选出diff>0.2*max(diff)的部分置零,其余为diff
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)  # 筛选出diff<delta的部分置零，其余为diff^2+delta^2
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        return diff


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def get_smooth_loss(disp, img, inputs=None, opts=None, gamma=1):
    """Computes the smoothness loss for a disp, The img is used for edge-aware smoothness
    """

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    if opts.use_gds_loss == 1:
        Mask_gr = 100 * inputs['dynamic_mask'] + (1 - inputs['dynamic_mask'])
        Mask_gr = F.interpolate(Mask_gr, size=(img.size(2), img.size(3)), mode='nearest')
        Mask_gr = Mask_gr[:, :, :-1, :]
        grad_disp_y = grad_disp_y * Mask_gr

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-gamma * grad_img_x)
    grad_disp_y *= torch.exp(-gamma * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.

    The loss can be described as:

    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(N, 3, 3)`.
        - Target: Shape :math:`(N, 3, 3)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        R_diffs = input @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()
