# Originally from megablocks
import torch
import stk
from megablocks import ops
from megablocks.ops import gather, scatter, padded_gather, padded_scatter
from grouped_gemm import backend


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
        b = b.to(a.dtype)
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


def topology(x, padded_bins, num_total_experts, hidden_dim, blocking):
    padded_tokens, _ = x.size()
    assert padded_tokens % blocking == 0
    assert hidden_size % blocking == 0

    # Offsets for the sparse matrix. All rows have the
    # same number of nonzero blocks dictated by the
    # dimensionality of a single expert.
    block_rows = padded_tokens // blocking
    blocks_per_row = hidden_dim // blocking
    offsets = torch.arange(
        0,
        block_rows * blocks_per_row + 1,
        blocks_per_row,
        dtype=torch.int32,
        device=x.device,
    )

    # Indices for the sparse matrix. The indices for
    # the intermediate matrix are dynamic depending
    # on the mapping of tokens to experts.
    column_indices = ops.topology(
        padded_bins,
        blocking,
        block_rows,
        blocks_per_row,
    )

    # TODO(tgale): This is unused. Remove the need for this in stk.
    # For now, use meta init to save the device memory.
    data = torch.empty(
        column_indices.numel(),
        blocking,
        blocking,
        dtype=x.dtype,
        device='meta',
    )
    shape = (
        padded_tokens,
        hidden_dim * num_total_experts,
    )
    row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)

    block_columns = shape[1] // blocking
    _, gather_indices = torch.sort(column_indices.int())
    column_indices_t = row_indices.gather(0, gather_indices.long())
    block_offsets_t = gather_indices.int()
    nnz_per_column = torch.histc(column_indices, block_columns)
    nnz_per_column = torch.cumsum(nnz_per_column, 0)
    offsets_t = torch.cat([
        torch.zeros((1,), dtype=torch.int32, device=row_indices.device),
        nnz_per_column,
    ])

    return stk.Matrix(
        shape,
        data,
        row_indices,
        column_indices,
        offsets,
        column_indices_t,
        offsets_t,
        block_offsets_t,
    )
