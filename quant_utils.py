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



def apply_quantization_to_layer_t(layer, quant_type: str) -> None:
    """
    In-place quantization of the parameters in the given layer using various 8-bit "types."
    quant_type can be one of ["int8", "nf8", "fp8_e4m3", "fp8_e5m2"] (or "none"/baseline).
    """
    if quant_type not in ["int8", "nf8", "fp8_e4m3", "fp8_e5m2"]:
        # Fallback or do nothing if 'none' or unrecognized
        return

    for param in layer.parameters():
        with torch.no_grad():
            param.copy_(quantize_tensor(param, quant_type))

def quantize_tensor_t(tensor: torch.Tensor, quant_type: str) -> torch.Tensor:
    """
    Wrapper that calls the appropriate quant function depending on quant_type.
    """
    if quant_type == "int8":
        return quantize_int8(tensor)
    elif quant_type == "nf8":
        return quantize_nf8(tensor)
    elif quant_type == "fp8_e4m3":
        return quantize_fp8(tensor, exponent_bits=4, mantissa_bits=3)
    elif quant_type == "fp8_e5m2":
        return quantize_fp8(tensor, exponent_bits=5, mantissa_bits=2)
    else:
        # no quant
        return tensor


# --------------------------
# 1) Standard int8
# --------------------------
def quantize_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Naive min-max int8 quantization with dequant (symmetric around min/max).
    """
    min_val = tensor.min()
    max_val = tensor.max()

    if min_val == max_val:
        # all elements the same
        return tensor

    qmin = 0
    qmax = 255  # 2^8 - 1
    scale = (max_val - min_val) / float(qmax)

    # Scale -> round -> clamp -> de-scale
    quantized = ((tensor - min_val) / scale).round().clamp(qmin, qmax)
    dequantized = quantized * scale + min_val
    return dequantized


# --------------------------
# 2) nf8 (NormalFloat8) [Toy version]
# --------------------------
def quantize_nf8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Illustrative rank-based "normal" quant:
      - Flatten & rank the tensor
      - Convert rank to a standard normal dev using erfinv
      - Then min-max scale that normal dev into 8-bit
      - Convert back to float
    This is *NOT* an exact match of QLoRA's NF4. Real NF4 is more sophisticated.
    """
    # Flatten
    flat = tensor.view(-1)
    n = flat.numel()

    # Edge case
    if n < 2:
        return tensor

    # 1) Sort to get ranks
    # ranks of each element in ascending order
    sorted_vals, sorted_idx = torch.sort(flat)
    # We'll create a rank array of same shape
    ranks = torch.zeros_like(sorted_idx, dtype=torch.float32)
    ranks[sorted_idx] = torch.arange(n, dtype=torch.float32, device=flat.device)

    # 2) Convert ranks -> fraction -> normal via erfinv
    # fraction in [0..1], shift to [-1..1]
    fraction = (ranks + 0.5) / n
    scaled_frac = 2.0 * fraction - 1.0
    # clamp to avoid erfinv domain errors
    scaled_frac = scaled_frac.clamp(-0.9999, 0.9999)
    normal_approx = torch.erfinv(scaled_frac)

    # 3) Now normal_approx is roughly N(0,1). We do min-max int8 quant on it
    #    or you can do some scale factor around e.g. [-3,3]
    min_n = normal_approx.min()
    max_n = normal_approx.max()

    qmin = 0
    qmax = 255
    scale = (max_n - min_n) / float(qmax)
    quantized = ((normal_approx - min_n) / scale).round().clamp(qmin, qmax)
    normal_deq = quantized * scale + min_n

    # 4) Map back into the shape of the original tensor
    #    This "destroys" the original values, of course, but that's the idea.
    out_tensor = normal_deq.view(tensor.shape)
    return out_tensor


# --------------------------
# 3) Basic FP8 (e4m3 / e5m2)
# --------------------------
def quantize_fp8(tensor: torch.Tensor, exponent_bits=4, mantissa_bits=3) -> torch.Tensor:
    """
    Approximate float->FP8 cast:
      - For each element, we do: sign, exponent, mantissa
      - We clamp exponent to representable range
      - We round mantissa
    Real hardware or libraries might do subnormals, etc. This is a simplified version.
    """
    # We'll handle zero/inf/nan in a naive way.

    # 1) Decompose the float
    # sign bit
    sign = torch.sign(tensor)
    sign[sign == 0] = 1  # treat 0 as positive sign

    abs_val = torch.abs(tensor)

    # 2) Get exponent in base-2
    # e = floor(log2(abs_val))
    # some trick with frexp?
    # frexp returns mantissa in [0.5,1) and exponent
    mantissa, exponent = torch.frexp(abs_val)
    # mantissa in [0.5,1) => we'll store that in a fixed number of bits

    # If abs_val=0 => frexp(0) => (0.0,0). Let's keep that as exponent=0, mantissa=0
    # we can do a mask for zeros
    zero_mask = (abs_val == 0)
    mantissa[zero_mask] = 0.0
    exponent[zero_mask] = 1 - (2 ** (1 - exponent_bits))  # minimal exponent?

    # 3) Range for exponent (biased form)
    # For a typical e4 => exponent can represent e.g. -8..+7. That means we have 2^(4)=16 exponent values
    # We'll do a "bias" approach: e.g. e4 => bias=7
    # e5 => bias=15, etc.
    # In e4m3 => exponent range is [-8..7], bias=7
    # In e5m2 => exponent range is [-16..15], bias=15
    bias = (2 ** (exponent_bits - 1)) - 1  # e.g. 4 bits => bias=7
    # clamp exponent
    e_min = -bias
    e_max = bias
    exponent = torch.clamp(exponent, e_min, e_max)

    # 4) Round mantissa to 'mantissa_bits'
    # mantissa is in [0.5,1) for normal values
    # We'll shift it so [1.0,2.0) => (like typical float representation?), then quant
    # But to keep it simpler: just do:
    #   scaled_m = round(mantissa * 2^mantissa_bits)
    #   mantissa_approx = scaled_m / 2^mantissa_bits
    #   re-shift if needed
    shift = 2 ** mantissa_bits
    scaled_m = (mantissa * shift).round().clamp(0, shift - 1)
    mantissa_approx = scaled_m / shift

    # reconstruct approximate absolute value
    # abs_approx = 2^exponent * mantissa_approx
    abs_approx = torch.ldexp(mantissa_approx, exponent)

    # reapply sign
    out = sign * abs_approx

    # mask out zeros
    out[zero_mask] = 0.0

    return out
