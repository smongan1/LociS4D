"""
Triton kernel for the sequential linear recurrence:

    h_i = a_i * h_{i-1} + x_i,   i = 1..L,   h_0 given as input

Shapes:
    x  : (B, L, D)   input at each step
    a  : (B, L, D)   elementwise decay/gate at each step
    h0 : (B, D)      initial state
    h  : (B, L, D)   output states h_1 .. h_L (h_0 is not included in the output,
                      but is used internally and is differentiable)

The recurrence itself is inherently sequential in time, so each Triton program
owns one (batch, feature-block) slice and walks over L with a plain for-loop.
Parallelism comes from spreading (B, D) across the grid, not from parallelizing
time. This is the right structure for "sequential state building" -- if you
later want to scale to very long L, look at a chunked/associative (Blelloch)
scan instead; happy to write that version too.

Backward pass (derivation):

  Let g_i := dL/dh_i (total gradient of the loss w.r.t. h_i, including the
  contribution that flows back through h_{i+1}, h_{i+2}, ... ).

  Since h_i = a_i * h_{i-1} + x_i, we have dh_{i+1}/dh_i = a_{i+1}, so

      g_i = grad_out_i + a_{i+1} * g_{i+1},        g_{L+1} := 0

  which is a *reverse* scan over the same recurrence structure. Once g_i is
  known:

      dL/dx_i  = g_i
      dL/da_i  = g_i * h_{i-1}
      dL/dh_0  = a_1 * g_1

  So backward is just the forward recurrence run in reverse with a_t and h_{t-1}
  swapped in for the appropriate roles.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Forward kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _fwd_kernel(
    x_ptr, a_ptr, h0_ptr, h_ptr,
    L, D,
    stride_xb, stride_xl, stride_xd,
    stride_ab, stride_al, stride_ad,
    stride_h0b, stride_h0d,
    stride_hb, stride_hl, stride_hd,
    BLOCK_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    h0_ptrs = h0_ptr + pid_b * stride_h0b + d_offsets * stride_h0d
    h_prev = tl.load(h0_ptrs, mask=d_mask, other=0.0).to(DTYPE)

    x_base = x_ptr + pid_b * stride_xb + d_offsets * stride_xd
    a_base = a_ptr + pid_b * stride_ab + d_offsets * stride_ad
    h_base = h_ptr + pid_b * stride_hb + d_offsets * stride_hd

    for t in range(0, L):
        x_t = tl.load(x_base + t * stride_xl, mask=d_mask, other=0.0).to(DTYPE)
        a_t = tl.load(a_base + t * stride_al, mask=d_mask, other=0.0).to(DTYPE)

        h_t = a_t * h_prev + x_t

        tl.store(h_base + t * stride_hl, h_t, mask=d_mask)
        h_prev = h_t


# --------------------------------------------------------------------------- #
# Backward kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _bwd_kernel(
    a_ptr, h_ptr, h0_ptr, grad_h_ptr,
    grad_x_ptr, grad_a_ptr, grad_h0_ptr,
    L, D,
    stride_ab, stride_al, stride_ad,
    stride_hb, stride_hl, stride_hd,
    stride_h0b, stride_h0d,
    stride_ghb, stride_ghl, stride_ghd,
    stride_gxb, stride_gxl, stride_gxd,
    stride_gab, stride_gal, stride_gad,
    stride_gh0b, stride_gh0d,
    BLOCK_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    a_base = a_ptr + pid_b * stride_ab + d_offsets * stride_ad
    h_base = h_ptr + pid_b * stride_hb + d_offsets * stride_hd
    gh_base = grad_h_ptr + pid_b * stride_ghb + d_offsets * stride_ghd
    gx_base = grad_x_ptr + pid_b * stride_gxb + d_offsets * stride_gxd
    ga_base = grad_a_ptr + pid_b * stride_gab + d_offsets * stride_gad

    h0_ptrs = h0_ptr + pid_b * stride_h0b + d_offsets * stride_h0d

    # g holds a_{t+2} * g_{t+2} carried from the previous (later) iteration;
    # starts at 0 since there is no time step after L.
    g = tl.zeros([BLOCK_D], dtype=DTYPE)

    for step in range(0, L):
        t = L - 1 - step  # walk backwards: t = L-1, L-2, ..., 0

        grad_out_t = tl.load(gh_base + t * stride_ghl, mask=d_mask, other=0.0).to(DTYPE)
        g = grad_out_t + g  # this is g_i for i = t+1

        # dL/dx_i = g_i
        tl.store(gx_base + t * stride_gxl, g, mask=d_mask)

        # h_{i-1}: h0 if t == 0, else h[t-1]
        h_prev = tl.where(
            t == 0,
            tl.load(h0_ptrs, mask=d_mask, other=0.0).to(DTYPE),
            tl.load(h_base + (t - 1) * stride_hl, mask=d_mask & (t > 0), other=0.0).to(DTYPE),
        )

        # dL/da_i = g_i * h_{i-1}
        tl.store(ga_base + t * stride_gal, g * h_prev, mask=d_mask)

        a_t = tl.load(a_base + t * stride_al, mask=d_mask, other=0.0).to(DTYPE)
        g = g * a_t  # becomes a_{t+1} * g_{t+1} for the next (earlier) iteration

    # after the loop g == a_1 * g_1 == dL/dh_0
    grad_h0_ptrs = grad_h0_ptr + pid_b * stride_gh0b + d_offsets * stride_gh0d
    tl.store(grad_h0_ptrs, g, mask=d_mask)


# --------------------------------------------------------------------------- #
# Complex-valued kernels
#
# Triton has no native complex dtype (that's the `KeyError: 'complex64'` you
# hit), so complex inputs are decomposed into separate contiguous real/imag
# float tensors and the complex multiply-add is done by hand:
#
#     (ar + i*ai) * (hr + i*hi) = (ar*hr - ai*hi) + i*(ar*hi + ai*hr)
#
# h_i = a_i*h_{i-1} + x_i is holomorphic in x, a, h0, but PyTorch's complex
# autograd convention still requires a conjugate at every multiplication:
# grad_input = grad_output * conj(d output / d input). Concretely:
#
#     g_i         = grad_out_i + conj(a_{i+1}) * g_{i+1}
#     grad_x_i    = g_i                    (coefficient is 1, conj(1)=1)
#     grad_a_i    = g_i * conj(h_{i-1})
#     grad_h0     = conj(a_1) * g_1
#
# (An earlier version of this file dropped the conjugates -- forward and
# grad_x matched the reference because their coefficients are real/1, which
# made the bug invisible until grad_a/grad_h0 were checked.)
# --------------------------------------------------------------------------- #
@triton.jit
def _fwd_kernel_complex(
    xr_ptr, xi_ptr, ar_ptr, ai_ptr, h0r_ptr, h0i_ptr, hr_ptr, hi_ptr,
    L, D,
    stride_xb, stride_xl, stride_xd,
    stride_ab, stride_al, stride_ad,
    stride_h0b, stride_h0d,
    stride_hb, stride_hl, stride_hd,
    BLOCK_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    h0r_ptrs = h0r_ptr + pid_b * stride_h0b + d_offsets * stride_h0d
    h0i_ptrs = h0i_ptr + pid_b * stride_h0b + d_offsets * stride_h0d
    hr_prev = tl.load(h0r_ptrs, mask=d_mask, other=0.0).to(DTYPE)
    hi_prev = tl.load(h0i_ptrs, mask=d_mask, other=0.0).to(DTYPE)

    xr_base = xr_ptr + pid_b * stride_xb + d_offsets * stride_xd
    xi_base = xi_ptr + pid_b * stride_xb + d_offsets * stride_xd
    ar_base = ar_ptr + pid_b * stride_ab + d_offsets * stride_ad
    ai_base = ai_ptr + pid_b * stride_ab + d_offsets * stride_ad
    hr_base = hr_ptr + pid_b * stride_hb + d_offsets * stride_hd
    hi_base = hi_ptr + pid_b * stride_hb + d_offsets * stride_hd

    for t in range(0, L):
        xr_t = tl.load(xr_base + t * stride_xl, mask=d_mask, other=0.0).to(DTYPE)
        xi_t = tl.load(xi_base + t * stride_xl, mask=d_mask, other=0.0).to(DTYPE)
        ar_t = tl.load(ar_base + t * stride_al, mask=d_mask, other=0.0).to(DTYPE)
        ai_t = tl.load(ai_base + t * stride_al, mask=d_mask, other=0.0).to(DTYPE)

        # complex multiply-add: h_t = a_t * h_prev + x_t
        hr_t = ar_t * hr_prev - ai_t * hi_prev + xr_t
        hi_t = ar_t * hi_prev + ai_t * hr_prev + xi_t

        tl.store(hr_base + t * stride_hl, hr_t, mask=d_mask)
        tl.store(hi_base + t * stride_hl, hi_t, mask=d_mask)

        hr_prev = hr_t
        hi_prev = hi_t


@triton.jit
def _bwd_kernel_complex(
    ar_ptr, ai_ptr, hr_ptr, hi_ptr, h0r_ptr, h0i_ptr,
    ghr_ptr, ghi_ptr,
    gxr_ptr, gxi_ptr, gar_ptr, gai_ptr, gh0r_ptr, gh0i_ptr,
    L, D,
    stride_ab, stride_al, stride_ad,
    stride_hb, stride_hl, stride_hd,
    stride_h0b, stride_h0d,
    stride_ghb, stride_ghl, stride_ghd,
    stride_gxb, stride_gxl, stride_gxd,
    stride_gab, stride_gal, stride_gad,
    stride_gh0b, stride_gh0d,
    BLOCK_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    ar_base = ar_ptr + pid_b * stride_ab + d_offsets * stride_ad
    ai_base = ai_ptr + pid_b * stride_ab + d_offsets * stride_ad
    hr_base = hr_ptr + pid_b * stride_hb + d_offsets * stride_hd
    hi_base = hi_ptr + pid_b * stride_hb + d_offsets * stride_hd
    ghr_base = ghr_ptr + pid_b * stride_ghb + d_offsets * stride_ghd
    ghi_base = ghi_ptr + pid_b * stride_ghb + d_offsets * stride_ghd
    gxr_base = gxr_ptr + pid_b * stride_gxb + d_offsets * stride_gxd
    gxi_base = gxi_ptr + pid_b * stride_gxb + d_offsets * stride_gxd
    gar_base = gar_ptr + pid_b * stride_gab + d_offsets * stride_gad
    gai_base = gai_ptr + pid_b * stride_gab + d_offsets * stride_gad

    h0r_ptrs = h0r_ptr + pid_b * stride_h0b + d_offsets * stride_h0d
    h0i_ptrs = h0i_ptr + pid_b * stride_h0b + d_offsets * stride_h0d

    gr = tl.zeros([BLOCK_D], dtype=DTYPE)
    gi = tl.zeros([BLOCK_D], dtype=DTYPE)

    for step in range(0, L):
        t = L - 1 - step

        gohr = tl.load(ghr_base + t * stride_ghl, mask=d_mask, other=0.0).to(DTYPE)
        gohi = tl.load(ghi_base + t * stride_ghl, mask=d_mask, other=0.0).to(DTYPE)
        gr = gohr + gr
        gi = gohi + gi

        tl.store(gxr_base + t * stride_gxl, gr, mask=d_mask)
        tl.store(gxi_base + t * stride_gxl, gi, mask=d_mask)

        hr_prev = tl.where(
            t == 0,
            tl.load(h0r_ptrs, mask=d_mask, other=0.0).to(DTYPE),
            tl.load(hr_base + (t - 1) * stride_hl, mask=d_mask & (t > 0), other=0.0).to(DTYPE),
        )
        hi_prev = tl.where(
            t == 0,
            tl.load(h0i_ptrs, mask=d_mask, other=0.0).to(DTYPE),
            tl.load(hi_base + (t - 1) * stride_hl, mask=d_mask & (t > 0), other=0.0).to(DTYPE),
        )

        # grad_a_i = g_i * conj(h_{i-1})   [PyTorch complex-autograd convention:
        # grad_input = grad_output * conj(d output / d input)]
        tl.store(gar_base + t * stride_gal, gr * hr_prev + gi * hi_prev, mask=d_mask)
        tl.store(gai_base + t * stride_gal, gi * hr_prev - gr * hi_prev, mask=d_mask)

        ar_t = tl.load(ar_base + t * stride_al, mask=d_mask, other=0.0).to(DTYPE)
        ai_t = tl.load(ai_base + t * stride_al, mask=d_mask, other=0.0).to(DTYPE)

        # g <- conj(a_t) * g, carried to the next (earlier) step
        new_gr = gr * ar_t + gi * ai_t
        new_gi = gi * ar_t - gr * ai_t
        gr = new_gr
        gi = new_gi

    gh0r_ptrs = gh0r_ptr + pid_b * stride_gh0b + d_offsets * stride_gh0d
    gh0i_ptrs = gh0i_ptr + pid_b * stride_gh0b + d_offsets * stride_gh0d
    tl.store(gh0r_ptrs, gr, mask=d_mask)
    tl.store(gh0i_ptrs, gi, mask=d_mask)


# --------------------------------------------------------------------------- #
# Autograd wrapper
# --------------------------------------------------------------------------- #
def _pick_block_d(D: int) -> int:
    return min(1024, triton.next_power_of_2(D))


def _real_compute_dtype(torch_dtype: torch.dtype):
    """float64 in -> compute in float64 (needed for gradcheck-level precision);
    everything else (fp16/bf16/fp32) -> compute in float32."""
    if torch_dtype == torch.float64:
        return torch.float64, tl.float64
    return torch.float32, tl.float32


def _complex_compute_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.complex128:
        return torch.float64, tl.float64
    return torch.float32, tl.float32


def _split_complex(t: torch.Tensor):
    """Contiguous complex tensor -> two contiguous real tensors (real, imag)."""
    t = t.contiguous()
    tr = torch.view_as_real(t)  # (..., 2), last dim: [real, imag]
    real = tr[..., 0].contiguous()
    imag = tr[..., 1].contiguous()
    return real, imag


class LinearRecurrenceFn(torch.autograd.Function):
    """
    h_i = a_i * h_{i-1} + x_i, for i = 1..L, with given h_0.

    Forward returns h with shape (B, L, D) containing h_1 .. h_L.
    Supports real (fp16/bf16/fp32) and complex (complex64/complex128) dtypes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, h0: torch.Tensor):
        assert x.shape == a.shape, "x and a must have the same (B, L, D) shape"
        assert x.dim() == 3, "x, a must be (B, L, D)"
        B, L, D = x.shape
        assert h0.shape == (B, D), "h0 must be (B, D)"
        assert x.is_cuda and a.is_cuda and h0.is_cuda, "inputs must be CUDA tensors"

        is_complex = x.is_complex()
        assert is_complex == a.is_complex() == h0.is_complex(), \
            "x, a, h0 must all be real or all be complex"

        ctx.is_complex = is_complex
        ctx.in_dtype_x = x.dtype
        ctx.in_dtype_a = a.dtype
        ctx.in_dtype_h0 = h0.dtype
        ctx.BLOCK_D = _pick_block_d(D)
        BLOCK_D = ctx.BLOCK_D
        grid = (B, triton.cdiv(D, BLOCK_D))

        if not is_complex:
            x = x.contiguous()
            a = a.contiguous()
            h0 = h0.contiguous()

            compute_dtype, tl_dtype = _real_compute_dtype(x.dtype)
            ctx.compute_dtype = compute_dtype
            ctx.tl_dtype = tl_dtype

            h = torch.empty((B, L, D), device=x.device, dtype=compute_dtype)

            _fwd_kernel[grid](
                x, a, h0, h,
                L, D,
                x.stride(0), x.stride(1), x.stride(2),
                a.stride(0), a.stride(1), a.stride(2),
                h0.stride(0), h0.stride(1),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_D=BLOCK_D,
                DTYPE=tl_dtype,
            )

            ctx.save_for_backward(a, h0, h)
            return h.to(x.dtype)

        else:
            compute_dtype, tl_dtype = _complex_compute_dtype(x.dtype)
            ctx.compute_dtype = compute_dtype
            ctx.tl_dtype = tl_dtype

            xr, xi = _split_complex(x)
            ar, ai = _split_complex(a)
            h0r, h0i = _split_complex(h0)

            hr = torch.empty((B, L, D), device=x.device, dtype=compute_dtype)
            hi = torch.empty((B, L, D), device=x.device, dtype=compute_dtype)

            _fwd_kernel_complex[grid](
                xr, xi, ar, ai, h0r, h0i, hr, hi,
                L, D,
                xr.stride(0), xr.stride(1), xr.stride(2),
                ar.stride(0), ar.stride(1), ar.stride(2),
                h0r.stride(0), h0r.stride(1),
                hr.stride(0), hr.stride(1), hr.stride(2),
                BLOCK_D=BLOCK_D,
                DTYPE=tl_dtype,
            )

            ctx.save_for_backward(ar, ai, hr, hi, h0r, h0i)
            h = torch.complex(hr, hi)
            return h.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor):
        BLOCK_D = ctx.BLOCK_D

        if not ctx.is_complex:
            a, h0, h = ctx.saved_tensors
            B, L, D = h.shape
            grad_h = grad_h.contiguous()
            compute_dtype, tl_dtype = ctx.compute_dtype, ctx.tl_dtype

            grad_x = torch.empty((B, L, D), device=h.device, dtype=compute_dtype)
            grad_a = torch.empty((B, L, D), device=h.device, dtype=compute_dtype)
            grad_h0 = torch.empty((B, D), device=h.device, dtype=compute_dtype)

            grid = (B, triton.cdiv(D, BLOCK_D))

            _bwd_kernel[grid](
                a, h, h0, grad_h,
                grad_x, grad_a, grad_h0,
                L, D,
                a.stride(0), a.stride(1), a.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                h0.stride(0), h0.stride(1),
                grad_h.stride(0), grad_h.stride(1), grad_h.stride(2),
                grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
                grad_a.stride(0), grad_a.stride(1), grad_a.stride(2),
                grad_h0.stride(0), grad_h0.stride(1),
                BLOCK_D=BLOCK_D,
                DTYPE=tl_dtype,
            )

            return (
                grad_x.to(ctx.in_dtype_x),
                grad_a.to(ctx.in_dtype_a),
                grad_h0.to(ctx.in_dtype_h0),
            )

        else:
            ar, ai, hr, hi, h0r, h0i = ctx.saved_tensors
            B, L, D = hr.shape
            compute_dtype, tl_dtype = ctx.compute_dtype, ctx.tl_dtype

            ghr, ghi = _split_complex(grad_h)

            gxr = torch.empty((B, L, D), device=hr.device, dtype=compute_dtype)
            gxi = torch.empty((B, L, D), device=hr.device, dtype=compute_dtype)
            gar = torch.empty((B, L, D), device=hr.device, dtype=compute_dtype)
            gai = torch.empty((B, L, D), device=hr.device, dtype=compute_dtype)
            gh0r = torch.empty((B, D), device=hr.device, dtype=compute_dtype)
            gh0i = torch.empty((B, D), device=hr.device, dtype=compute_dtype)

            grid = (B, triton.cdiv(D, BLOCK_D))

            _bwd_kernel_complex[grid](
                ar, ai, hr, hi, h0r, h0i,
                ghr, ghi,
                gxr, gxi, gar, gai, gh0r, gh0i,
                L, D,
                ar.stride(0), ar.stride(1), ar.stride(2),
                hr.stride(0), hr.stride(1), hr.stride(2),
                h0r.stride(0), h0r.stride(1),
                ghr.stride(0), ghr.stride(1), ghr.stride(2),
                gxr.stride(0), gxr.stride(1), gxr.stride(2),
                gar.stride(0), gar.stride(1), gar.stride(2),
                gh0r.stride(0), gh0r.stride(1),
                BLOCK_D=BLOCK_D,
                DTYPE=tl_dtype,
            )

            grad_x = torch.complex(gxr, gxi).to(ctx.in_dtype_x)
            grad_a = torch.complex(gar, gai).to(ctx.in_dtype_a)
            grad_h0 = torch.complex(gh0r, gh0i).to(ctx.in_dtype_h0)
            return grad_x, grad_a, grad_h0


def linear_recurrence(x: torch.Tensor, a: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """
    h_i = a_i * h_{i-1} + x_i for i = 1..L, given h_0.

    Args:
        x:  (B, L, D)
        a:  (B, L, D)
        h0: (B, D)
    Returns:
        h:  (B, L, D)  containing h_1 .. h_L
    """
    return LinearRecurrenceFn.apply(x, a, h0)


# --------------------------------------------------------------------------- #
# Reference implementation + self-test
# --------------------------------------------------------------------------- #
def linear_recurrence_reference(x: torch.Tensor, a: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    h_prev = h0
    outs = []
    for t in range(L):
        h_t = a[:, t, :] * h_prev + x[:, t, :]
        outs.append(h_t)
        h_prev = h_t
    return torch.stack(outs, dim=1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available here -- run this on a GPU machine to test.")
    else:
        torch.manual_seed(0)
        B, L, D = 4, 37, 96
        device = "cuda"

        x = torch.randn(B, L, D, device=device, dtype=torch.float32).requires_grad_()
        a = (torch.rand(B, L, D, device=device, dtype=torch.float32) * 0.9 + 0.05).requires_grad_()
        h0 = torch.randn(B, D, device=device, dtype=torch.float32).requires_grad_()

        x_ref = x.detach().clone().requires_grad_()
        a_ref = a.detach().clone().requires_grad_()
        h0_ref = h0.detach().clone().requires_grad_()

        h_triton = linear_recurrence(x, a, h0)
        h_ref = linear_recurrence_reference(x_ref, a_ref, h0_ref)

        print("forward max abs diff:", (h_triton - h_ref).abs().max().item())

        grad_out = torch.randn_like(h_triton)
        h_triton.backward(grad_out)
        h_ref.backward(grad_out)

        print("grad_x  max abs diff:", (x.grad - x_ref.grad).abs().max().item())
        print("grad_a  max abs diff:", (a.grad - a_ref.grad).abs().max().item())
        print("grad_h0 max abs diff:", (h0.grad - h0_ref.grad).abs().max().item())

        # gradcheck against the reference (finite differences on the reference
        # implementation is generally what you'd sanity check the kernel
        # against; here we double-check the reference itself against autograd
        # for extra confidence, then rely on the diffs above).
        from torch.autograd import gradcheck

        xd = torch.randn(2, 5, 8, device=device, dtype=torch.float64).requires_grad_()
        ad = (torch.rand(2, 5, 8, device=device, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()
        h0d = torch.randn(2, 8, device=device, dtype=torch.float64).requires_grad_()
        print("gradcheck vs analytic (double precision, small case):",
              gradcheck(lambda xx, aa, hh: linear_recurrence(xx, aa, hh), (xd, ad, h0d), eps=1e-6, atol=1e-4))

        # --- complex64 path -------------------------------------------------
        Bc, Lc, Dc = 3, 17, 32
        xc = torch.randn(Bc, Lc, Dc, device=device, dtype=torch.complex64).requires_grad_()
        # keep |a| < 1 for stability, random phase
        mag = torch.rand(Bc, Lc, Dc, device=device) * 0.9 + 0.05
        phase = torch.rand(Bc, Lc, Dc, device=device) * 2 * torch.pi
        ac = (mag * torch.exp(1j * phase)).to(torch.complex64).requires_grad_()
        h0c = torch.randn(Bc, Dc, device=device, dtype=torch.complex64).requires_grad_()

        xc_ref = xc.detach().clone().requires_grad_()
        ac_ref = ac.detach().clone().requires_grad_()
        h0c_ref = h0c.detach().clone().requires_grad_()

        hc_triton = linear_recurrence(xc, ac, h0c)
        hc_ref = linear_recurrence_reference(xc_ref, ac_ref, h0c_ref)
        print("complex forward max abs diff:", (hc_triton - hc_ref).abs().max().item())

        grad_out_c = torch.randn_like(hc_triton)
        hc_triton.backward(grad_out_c)
        hc_ref.backward(grad_out_c)

        print("complex grad_x  max abs diff:", (xc.grad - xc_ref.grad).abs().max().item())
        print("complex grad_a  max abs diff:", (ac.grad - ac_ref.grad).abs().max().item())
        print("complex grad_h0 max abs diff:", (h0c.grad - h0c_ref.grad).abs().max().item())

        xcd = torch.randn(2, 5, 6, device=device, dtype=torch.complex128).requires_grad_()
        magd = torch.rand(2, 5, 6, device=device, dtype=torch.float64) * 0.8 + 0.1
        phased = torch.rand(2, 5, 6, device=device, dtype=torch.float64) * 2 * torch.pi
        acd = (magd * torch.exp(1j * phased)).to(torch.complex128).requires_grad_()
        h0cd = torch.randn(2, 6, device=device, dtype=torch.complex128).requires_grad_()
        print("complex gradcheck (double precision, small case):",
              gradcheck(lambda xx, aa, hh: linear_recurrence(xx, aa, hh), (xcd, acd, h0cd), eps=1e-6, atol=1e-4))