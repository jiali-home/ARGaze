import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3tuple(value):
    if isinstance(value, tuple):
        if len(value) == 3:
            return value
        if len(value) == 1:
            return (value[0], value[0], value[0])
        raise ValueError("Padding/stride/dilation tuple must have length 3.")
    return (value, value, value)


class TemporalCausalConv3d(nn.Conv3d):
    """Depthwise/grouped Conv3d with causal padding along the temporal axis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_pad = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x):
        if self.temporal_pad > 0:
            pad = (0, 0, 0, 0, self.temporal_pad, 0)
            x = F.pad(x, pad)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @classmethod
    def from_conv(cls, conv: nn.Conv3d) -> "TemporalCausalConv3d":
        padding = _to_3tuple(conv.padding)
        spatial_padding = (0, padding[1], padding[2])
        causal_conv = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=spatial_padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        causal_conv.load_state_dict(conv.state_dict())
        return causal_conv


def convert_conv3d_to_causal(conv: nn.Module):
    """Return a causally padded Conv3d if possible."""
    if conv is None or not isinstance(conv, nn.Conv3d):
        return conv
    if isinstance(conv, TemporalCausalConv3d):
        return conv
    kernel = _to_3tuple(conv.kernel_size)
    if kernel[0] <= 1:
        return conv
    return TemporalCausalConv3d.from_conv(conv)


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def build_temporal_causal_mask(
    q_shape,
    k_shape,
    has_cls_embed,
    has_global_embed,
    global_embed_num,
    device,
    dtype,
):
    """Create an additive causal mask for attention logits."""
    num_global = global_embed_num if has_global_embed else 0
    num_cls = 1 if has_cls_embed else 0
    q_patches = _numel(q_shape)
    k_patches = _numel(k_shape)
    total_q = num_global + num_cls + q_patches
    total_k = num_global + num_cls + k_patches
    mask = torch.zeros(total_q, total_k, device=device, dtype=dtype)

    if q_patches == 0 or k_patches == 0:
        return mask

    Tq = max(q_shape[0], 1)
    Tk = max(k_shape[0], 1)
    q_hw = max(q_shape[1] * q_shape[2], 1)
    k_hw = max(k_shape[1] * k_shape[2], 1)

    q_time_idx = torch.arange(q_patches, device=device, dtype=torch.long) // q_hw
    k_time_idx = torch.arange(k_patches, device=device, dtype=torch.long) // k_hw

    numerator = (q_time_idx + 1) * Tk - 1
    allowed_k_idx = torch.div(numerator, Tq, rounding_mode="floor")
    allowed_k_idx = torch.clamp(allowed_k_idx, min=0, max=Tk - 1)

    causal = k_time_idx.unsqueeze(0) <= allowed_k_idx.unsqueeze(1)
    neg_inf = torch.finfo(dtype).min
    patch_region = mask[num_global + num_cls :, num_global + num_cls :]
    patch_region[~causal] = neg_inf
    return mask
