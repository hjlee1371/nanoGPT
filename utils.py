import torch

class MoEAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, aux_loss):
        ctx.save_for_backward(aux_loss)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        aux_loss, = ctx.saved_tensors
        aux_loss_grad = torch.ones_like(aux_loss)
        return grad_output, aux_loss_grad

apply_aux_loss = MoEAuxLoss.apply
