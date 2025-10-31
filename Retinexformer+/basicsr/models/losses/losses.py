import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')




class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class CosineLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super(CosineLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        # Flatten to [B, -1]
        input_flat = input.view(input.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Normalize
        input_norm = F.normalize(input, dim=-1, eps=self.eps)
        target_norm = F.normalize(target, dim=-1, eps=self.eps)

        # Cosine similarity
        cos_sim = (input_norm * target_norm).sum(dim=-1)
        cos_loss = 1.0 - cos_sim  # Cosine distance

        if self.reduction == 'mean':
            return cos_loss.mean()
        elif self.reduction == 'sum':
            return cos_loss.sum()
        else:
            return cos_loss  # [B


class WHTBlock(nn.Module):
    def __init__(self, block_size=16, thresh=0.1,calc_iwht=True,isdiff=True,final_level='wht',normalized=True,updown_flg=False):
        'thresh=0.1 or (0,0.1,0.2)'
        super(WHTBlock, self).__init__()
        self.walsh_matrix = self.generate_walsh_matrix(block_size)
        # self.wht_matrix,self.wht_matrixT = self.generate_wht_matrix(self.wht2d_matrix)
        self.block_size = block_size
        self.normalized = normalized
        self.calc_iwht=calc_iwht
        self.threshold=thresh
        self.isdiff = isdiff
        self.final_level = final_level #'ori' 'wht'
        self.updown_flg = updown_flg  #
    @staticmethod
    def generate_walsh_matrix(n):
        """
        递归生成沃尔什-哈达玛矩阵
        参数:
            n (int): 必须是2的幂次
        返回:
            torch.Tensor: (n, n) 的沃尔什矩阵
        """
        assert (n & (n - 1)) == 0, "n必须是2的幂次"

        if n == 1:
            return torch.ones((1, 1))

        h = WHTBlock.generate_walsh_matrix(n // 2)
        return torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1)
        ], dim=0)

    def wht_1d(self, x):
        """
        一维沃尔什-哈达玛变换
        参数:
            x (torch.Tensor): (..., n)
        返回:
            torch.Tensor: (..., n)
        """
        n = x.shape[-1]
        # H = self.generate_walsh_matrix(n).to(x.device)
        H = self.walsh_matrix.to(x.device)
        if self.normalized:
            H = H / np.sqrt(n)

        return torch.matmul(x, H.T)

    def iwht_1d(self, x):
        """
        一维逆沃尔什-哈达玛变换
        参数:
            x (torch.Tensor): (..., n)
        返回:
            torch.Tensor: (..., n)
        """
        n = x.shape[-1]
        # H = self.generate_walsh_matrix(n).to(x.device)
        H = self.walsh_matrix.to(x.device)
        if self.normalized:
            H = H.T / np.sqrt(n)
        else:
            H = H.T / n

        return torch.matmul(x, H)

    def wht_2d(self, x):
        """
        二维沃尔什-哈达玛变换
        参数:
            x (torch.Tensor): (..., h, w)
        返回:
            torch.Tensor: (..., h, w)
        """
        # 行变换
        x = self.wht_1d(x)
        # 列变换
        x = self.wht_1d(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    def iwht_2d(self, x):
        """
        二维逆沃尔什-哈达玛变换
        参数:
            x (torch.Tensor): (..., h, w)
        返回:
            torch.Tensor: (..., h, w)
        """
        # 行逆变换
        x = self.iwht_1d(x)
        # 列逆变换
        x = self.iwht_1d(x.transpose(-1, -2)).transpose(-1, -2)
        return x
    def call0(self, input,thresh=0.1):
        b,c,h,w = input.shape
        block_size = self.block_size
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        input_ = F.pad(input, (0, w_pad, 0, h_pad), 'reflect')
        # print(f'b: {b}, c: {c}, h: {h}, w: {w}, h_pad: {h_pad}, w_pad: {w_pad},final: {input_.shape}')
        # h_new, w_new = input_.shape[-2:]
        # col_num = w_new // block_size
        # row_num = h_new // block_size
        # print(f'col_num: {col_num}, row_num: {row_num}')
        input_reshape = input_.unfold(2,block_size, block_size).unfold(3,block_size,block_size)
        x = self.wht_2d(input_reshape)
        x_coeff = torch.abs(x)
        x[ x_coeff < 0.02 ] = 0
        y = self.iwht_2d(x)
        y00 = draw_img(y)
        show_img(input_,y00,showFlg=True)
        return x,y
    def cal_wht(self,input,threshold):
        b, c, h, w = input.shape
        block_size = self.block_size
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        input_ = F.pad(input, (0, w_pad, 0, h_pad), 'reflect')
        # print(f'b: {b}, c: {c}, h: {h}, w: {w}, h_pad: {h_pad}, w_pad: {w_pad},final: {input_.shape}')
        h_new, w_new = input_.shape[-2:]
        # col_num = w_new // block_size
        # row_num = h_new // block_size
        # print(f'col_num: {col_num}, row_num: {row_num}')
        input_reshape = input_.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        x = self.wht_2d(input_reshape)
        x_coeff = torch.abs(x)
        x_out_list=[]
        if isinstance(threshold,(int,float)):
            x_out = x
            x_out[x_coeff < threshold] = 0
            x_out_list.append(x_out)
            return x_out_list
        elif isinstance(threshold, (np.ndarray, list, torch.Tensor)):  # 如果 threshold 是数组
            if isinstance(threshold, torch.Tensor):  # 转换 threshold 为 Tensor，确保它是一致的类型
                threshold = threshold.numpy()  # 转换为 numpy 数组，便于广播
            # 如果是数组，则输出多个矩阵
            # output_matrices = []
            for t in threshold:
                x_copy = x.clone()  # 克隆一份 x
                x_copy[x_coeff < t] = 0
                x_out_list.append(x_copy)
            return x_out_list  # 返回多个矩阵
        else:
            raise ValueError("Threshold should be either a number or an array.")
    def cal_wht_2d_diff(self,input,threshold,isdiff=True,final_level='ori'):
        b, c, h, w = input.shape
        block_size = self.block_size
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        input_ = F.pad(input, (0, w_pad, 0, h_pad), 'reflect')
        # print(f'b: {b}, c: {c}, h: {h}, w: {w}, h_pad: {h_pad}, w_pad: {w_pad},final: {input_.shape}')
        h_new, w_new = input_.shape[-2:]
        # col_num = w_new // block_size
        # row_num = h_new // block_size
        # print(f'col_num: {col_num}, row_num: {row_num}')
        input_reshape = input_.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        x = self.wht_2d(input_reshape)
        x_coeff = torch.abs(x)
        y_copy_list = []
        if isinstance(threshold, (int, float)):
            x_copy = x.clone()  # 克隆一份 x
            x_copy[x_coeff < threshold] = 0
            y_copy = self.iwht_2d(x_copy)
            if isdiff:
                y_copy_list.append(torch.abs(input_reshape - y_copy))
            else:
                y_copy_list.append(y_copy)
        elif isinstance(threshold, (np.ndarray, list, torch.Tensor)):  # 如果 threshold 是数组
            if isinstance(threshold, torch.Tensor):  # 转换 threshold 为 Tensor，确保它是一致的类型
                threshold = threshold.numpy()  # 转换为 numpy 数组，便于广播
            # 如果是数组，则输出多个矩阵
            # output_matrices = []
            for t in threshold:
                x_copy = x.clone()  # 克隆一份 x
                x_copy[x_coeff < t] = 0
                y_copy = self.iwht_2d(x_copy)
                if isdiff:
                    y_copy_list.append(torch.abs(input_reshape - y_copy))
                else:
                    y_copy_list.append(y_copy)
        else:
            raise ValueError("Threshold should be either a number or an array.")
        if final_level == 'ori':
            y_copy_list.append(input_reshape)
        elif final_level == 'wht':
            y_copy_list.append(x)
        return y_copy_list
    def call_half_wht(self, input):
        b,c,h,w = input.shape
        block_size = self.block_size
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        input_ = F.pad(input, (0, w_pad, 0, h_pad), 'reflect')
        input_reshape = input_.unfold(2,block_size, block_size).unfold(3,block_size,block_size)
        x = self.wht_2d(input_reshape)
        x_coeff_up = x.clone()
        x_coeff_down = x.clone()
        x_coeff_down[:,:,:,:,:block_size//2,:block_size//2] = 0
        x_coeff_up[:,:,:,:,block_size // 2:,:block_size//2] = 0
        y_copy_list = []
        y_up = self.iwht_2d(x_coeff_up)
        y_down = self.iwht_2d(x_coeff_down)
        y = self.iwht_2d(x)
        y_copy_list.append(y_up)
        y_copy_list.append(y_down)
        y_copy_list.append(x)
        return  y_copy_list
    def forward(self, input):
        if self.updown_flg:
            outlist = self.call_half_wht(input)
            return outlist
        if self.calc_iwht:
            outlist = self.cal_wht(input, self.threshold)
        else:
            outlist = self.cal_wht_2d_diff(input=input, threshold=self.threshold, isdiff=self.isdiff,
                                           final_level=self.final_level)
        return outlist



class SASWLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SASWLoss,self).__init__()
        self.block_size = 32 #8 16 32 64 128
        self.thresh = [0.2,0.1,0.05] # [0.2,0.1,0.05]
        # self.wht = WHT(self.block_size)
        # 'final_level='ori' 'wht'
        self.wht = WHTBlock(block_size=self.block_size, thresh=self.thresh, calc_iwht=True, isdiff=True, final_level='wht',
                            normalized=True,updown_flg=True)
        self.lambda_value = [1, 2, 1, 1]  # if not use iwht weight need multipy self.block_size
        self.l1_loss = nn.L1Loss(reduction='mean')
        # self.cosloss = CosineLoss()
        # self.loss1 = nn.HuberLoss()
        self.loss1 = CosineLoss()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
    def forward(self,pred,gt,weight=None, **kwargs):
        pred_list = self.wht(pred)
        gt_list = self.wht(gt)
        assert len(pred_list) == len(gt_list)
        loss_val = 0.0
        for i in range(len(pred_list) - 1):
            if i ==1:
                pred_level = pred_list[i]
                gt_level = gt_list[i]
                assert pred_level.shape == gt_level.shape, f"Layer {i} shape mismatch"
                # loss_val += self.lambda_value[i] * torch.mean(torch.log1p(torch.abs(pred_level-gt_level)))
                loss_val += self.lambda_value[i] * self.loss1(pred_level.reshape(pred_level.size(0),-1),gt_level.reshape(pred_level.size(0),-1))

        pred_level = pred_list[-1]
        gt_level = gt_list[-1]
        loss_val += self.lambda_value[-1] * torch.mean(self.l1_loss(pred_level, gt_level))
        return loss_val
