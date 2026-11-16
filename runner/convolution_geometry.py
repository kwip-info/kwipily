"""Shared allocation-free geometry for a single validated convolution axis."""


def resolve_axis(length, kernel, stride, dilation, padding, policy, mode, *, operation):
    """Arguments are canonical configuration integers; caller validates sample rank/channels."""
    if not 1 <= length <= 4096:
        raise ValueError(f'{operation} input length must be from 1 to 4096 on each axis.')
    effective_kernel = dilation * (kernel - 1) + 1
    if policy == 'explicit':
        left = right = padding
    elif policy == 'valid':
        left = right = 0
    elif policy == 'same':
        total = effective_kernel - 1
        left = total // 2
        right = total - left
    else:
        raise ValueError(f'{operation} has an unsupported padding policy.')
    if left > 4096 or right > 4096:
        raise ValueError(f'{operation} padding may not exceed 4096 on either side.')
    if mode == 'reflect' and (left >= length or right >= length):
        raise ValueError(f'{operation} reflection padding must be smaller than input length on each side.')
    if mode == 'circular' and (left > length or right > length):
        raise ValueError(f'{operation} circular padding cannot exceed input length on either side.')
    output = (length + left + right - effective_kernel) // stride + 1
    if not 1 <= output <= 4096:
        raise ValueError(f'{operation} output length must be from 1 to 4096; review kernel, dilation, stride and padding.')
    return output, left, right
