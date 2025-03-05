import logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
# GPT-2 "Conv1D" might live in different places depending on Transformers version
from transformers.modeling_utils import Conv1D

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {"nf4", "fp4", "int8", "fp16", "bf16", "fp32"}
Q_TYPE_SIZE = {"nf4": 4, "fp4": 4, "int8": 8, "fp16": 16, "bf16": 16, "fp32": 32}


class Quantizer:
    def __init__(
            self,
            compute_dtype: torch.dtype = torch.float16,
            compress_statistics: bool = True,
            quant_storage: torch.dtype = torch.uint8,
            target_device: torch.device = None,
    ):
        """
        Args:
            compute_dtype: For bitsandbytes 4-bit layers (e.g. torch.float16, torch.float32).
            compress_statistics: bnb 4-bit argument controlling whether to compress group statistics.
            quant_storage: Usually torch.uint8 for 4-bit data.
            target_device: CPU or CUDA device to place new layers on.
        """
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.quant_storage = quant_storage
        if target_device is None:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_device = target_device

    def quantize_linear_layer(self, float_linear: nn.Module, quant_type: str) -> nn.Module:
        """
        Replaces a GPT-2 Conv1D or PyTorch nn.Linear with a quantized/typed linear layer.

        - If `float_linear` is Conv1D with weight [in_dim, out_dim],
          we transpose -> [out_dim, in_dim] to match nn.Linear’s usual shape.
        - If `float_linear` is nn.Linear, we use its existing shape as is.
        - Then we build a bitsandbytes quantized or standard nn.Linear in the requested quant_type,
          copy the float weights/bias, and do a final `_quantize()` if 4-bit.

        Returns:
            nn.Module: The new layer with shape [out_features, in_features], ready to run.
        """
        if quant_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported quant_type='{quant_type}'. Must be one of {SUPPORTED_TYPES}")

        # Ensure the source module is on correct device
        float_linear = float_linear.to(self.target_device)

        # 1) If GPT-2 Conv1D, convert to a normal float linear
        if isinstance(float_linear, Conv1D):
            old_weight = float_linear.weight  # shape [in_dim, out_dim]
            old_bias = float_linear.bias
            has_bias = (old_bias is not None)

            in_features = old_weight.shape[0]
            out_features = old_weight.shape[1]

            # Make a float linear: shape => [out_features, in_features]
            tmp_linear = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
                device=self.target_device,
                dtype=torch.float32  # you could do float16 here, but float32 is safer prior to quant
            )
            with torch.no_grad():
                tmp_linear.weight.copy_(old_weight.T)  # transpose
                if has_bias:
                    tmp_linear.bias.copy_(old_bias)

            float_linear = tmp_linear

        elif isinstance(float_linear, nn.Linear):
            in_features = float_linear.in_features
            out_features = float_linear.out_features
            has_bias = (float_linear.bias is not None)

        else:
            raise ValueError(f"quantize_linear_layer: unsupported layer type {type(float_linear)}. "
                             f"Expected Conv1D or nn.Linear.")

        # 2) Build the quantized or typed layer
        if quant_type in ("nf4", "fp4"):
            # BitsAndBytes 4-bit
            new_layer = bnb.nn.Linear4bit(
                input_features=in_features,
                output_features=out_features,
                bias=has_bias,
                compute_dtype=self.compute_dtype,
                compress_statistics=self.compress_statistics,
                quant_type=quant_type,
                quant_storage=self.quant_storage,
                device=self.target_device
            )
        elif quant_type == "int8":
            # BitsAndBytes 8-bit
            new_layer = bnb.nn.Linear8bitLt(
                input_features=in_features,
                output_features=out_features,
                bias=has_bias,
                device=self.target_device,
            )
        else:
            # "fp16", "bf16", or "fp32" => standard PyTorch
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            if quant_type not in dtype_map:
                raise ValueError(f"Internal error: unhandled quant_type '{quant_type}' in else-branch.")
            target_dtype = dtype_map[quant_type]
            new_layer = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
                device=self.target_device,
                dtype=target_dtype
            )

        # 3) Copy the float weights into the new layer
        with torch.no_grad():
            new_layer.weight.copy_(
                float_linear.weight.to(new_layer.weight.dtype)
            )
            if has_bias:
                new_layer.bias.copy_(
                    float_linear.bias.to(new_layer.bias.dtype)
                )

        # 4) For NF4/FP4, do final quantization
        if quant_type in ("nf4", "fp4"):
            new_layer.weight._quantize(device=self.target_device)

        return new_layer


class MixedQuantizer:
    """
    Applies a repeating "quant_schema" across all layers in 'model.layers'.
    Or you can adapt it to any other attribute that is a list of submodules.
    """

    def __init__(self, quant_schema, quantizer: Quantizer):
        self.quant_schema = quant_schema
        self.quantizer = quantizer

    def quantize_model(self, model: nn.Module, layer_attribute: str = "model.layers") -> nn.Module:
        layers = getattr(model, layer_attribute.split(".")[0])
        for attr in layer_attribute.split(".")[1:]:
            layers = getattr(layers, attr)

        num_layers = len(layers)
        # Repeat the schema enough times to cover all layers
        schema = (self.quant_schema * ((num_layers + len(self.quant_schema) - 1) // len(self.quant_schema)))[
                 :num_layers]

        for layer, qtype in zip(layers, schema):
            self._quantize_layer(layer, qtype)
        return model

    def _quantize_layer(self, module: nn.Module, quant_type: str):
        # Recursively traverse submodules. If you find nn.Linear or Conv1D, replace it.
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, Conv1D)):
                logger.info(f"Quantizing layer {name} with type {quant_type}")
                new_mod = self.quantizer.quantize_linear_layer(child, quant_type)
                setattr(module, name, new_mod)
            else:
                self._quantize_layer(child, quant_type)
