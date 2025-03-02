import logging
import torch
import torch.nn as nn
import bitsandbytes as bnb

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {"nf4", "fp4", "int8", "fp16", "bf16", "fp32"}

Q_TYPE_SIZE = {"nf4": 4, "fp4": 4, "int8": 8, "fp16": 16, "bf16": 16, "fp32": 32}


class Quantizer:
    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        compress_statistics: bool = True,
        quant_storage: torch.dtype = torch.uint8,
    ):
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.quant_storage = quant_storage

    def quantize_linear_layer(self, float_linear: nn.Linear, quant_type: str) -> nn.Module:
        if quant_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported quant_type='{quant_type}'. Supported: {SUPPORTED_TYPES}")

        in_features = float_linear.in_features
        out_features = float_linear.out_features
        has_bias = float_linear.bias is not None
        device = float_linear.weight.device

        if quant_type in ("nf4", "fp4"):
            new_layer = bnb.nn.Linear4bit(
                input_features=in_features,
                output_features=out_features,
                bias=has_bias,
                compute_dtype=self.compute_dtype,
                compress_statistics=self.compress_statistics,
                quant_type=quant_type,
                quant_storage=self.quant_storage,
                device=device,
            )
        elif quant_type == "int8":
            new_layer = bnb.nn.Linear8bitLt(
                input_features=in_features,
                output_features=out_features,
                bias=has_bias,
                device=device,
            )
        else:  # PyTorch Linear layers
            dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32
            }[quant_type]
            new_layer = nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype)

        # Safe copying parameters
        with torch.no_grad():
            new_layer.weight.copy_(float_linear.weight.to(new_layer.weight.dtype))
            if has_bias:
                new_layer.bias.copy_(float_linear.bias.to(new_layer.bias.dtype))

        return new_layer

class MixedQuantizer:
    def __init__(self, quant_schema, quantizer: Quantizer):
        self.quant_schema = quant_schema
        self.quantizer = quantizer

    def quantize_model(self, model: nn.Module, layer_attribute: str = "model.layers") -> nn.Module:
        layers = getattr(model, layer_attribute.split(".")[0])
        for attr in layer_attribute.split(".")[1:]:
            layers = getattr(layers, attr)

        num_layers = len(layers)
        schema = (self.quant_schema * ((num_layers + len(self.quant_schema) - 1) // len(self.quant_schema)))[:num_layers]

        for layer, qtype in zip(layers, schema):
            self._quantize_layer(layer, qtype)
        return model

    def _quantize_layer(self, module: nn.Module, quant_type: str):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, self.quantizer.quantize_linear_layer(child, quant_type))
            else:
                self._quantize_layer(child, quant_type)
