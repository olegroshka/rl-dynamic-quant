# quant_utils.py

import torch

def quantize_tensor(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Quantize a tensor to the specified number of bits using naive min-max scaling.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    if min_val == max_val:
        # Edge case: constant tensor
        return tensor

    qmin = 0
    qmax = 2 ** num_bits - 1
    scale = (max_val - min_val) / float(qmax)

    # Scale tensor to [0, qmax], round, then scale back
    quantized = ((tensor - min_val) / scale).round().clamp(qmin, qmax)
    dequantized = quantized * scale + min_val
    return dequantized


def apply_quantization_to_layer(layer, num_bits: int) -> None:
    """
    In-place quantization of the parameters in the given layer to `num_bits`.
    """
    for param in layer.parameters():
        with torch.no_grad():
            param.copy_(quantize_tensor(param, num_bits))
