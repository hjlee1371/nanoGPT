from grouped_gemm import backend
import torch


@torch.library.custom_op("grouped_gemm::wrapped_gmm", mutates_args=())
def wrapped_gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    return backend.gmm(a, b, batch_sizes, trans_a=trans_a, trans_b=trans_b)

@wrapped_gmm.register_fake
def wrapped_gmm_fake(a, b, batch_sizes, trans_a, trans_b):
    if trans_a:
        shape = (batch_sizes.shape[0], a.shape[1], b.shape[1])
    else:
        shape = (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    return a.new_empty(*shape)

class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return wrapped_gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = wrapped_gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = wrapped_gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)
