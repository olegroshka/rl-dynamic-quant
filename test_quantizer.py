# test_quantizer.py

import unittest
import torch
import bitsandbytes as bnb
from transformers.modeling_utils import Conv1D
from quantizer import Quantizer, MixedQuantizer, SUPPORTED_TYPES

class NestedMockModel(torch.nn.Module):
    """
    Fake model with nested layers for testing.
    """
    def __init__(self, n_layers=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([MockLayer() for _ in range(n_layers)])
        self.model.layers[0].fc1 = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.Linear(8, 8)
        )

class MockLayer(torch.nn.Module):
    """
    Simple module with two Linear submodules for testing.
    """
    def __init__(self, in_f=8, out_f=8):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_f, out_f, bias=True)
        self.fc2 = torch.nn.Linear(in_f, out_f, bias=True)

class MockModel(torch.nn.Module):
    """
    Fake Transformer-like model with .model.layers = ModuleList([...]).
    """
    def __init__(self, n_layers=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([MockLayer() for _ in range(n_layers)])

class GPT2MockModel(torch.nn.Module):
    """
    Fake GPT-2 model with Conv1D layers for testing.
    """
    def __init__(self, n_layers=3):
        super().__init__()
        self.h = torch.nn.ModuleList([GPT2MockLayer() for _ in range(n_layers)])

class GPT2MockLayer(torch.nn.Module):
    """
    Fake GPT-2 layer with Conv1D layers for testing.
    """
    def __init__(self):
        super().__init__()
        self.c_attn = Conv1D(768, 2304)  # Query, Key, Value projections
        self.c_proj = Conv1D(768, 768)   # Output projection
        self.c_fc = Conv1D(768, 3072)    # Feed-forward intermediate
        self.c_proj2 = Conv1D(3072, 768) # Feed-forward output

class TestQuantizer(unittest.TestCase):
    def setUp(self):
        self.quantizer = Quantizer()
        self.in_features = 8
        self.out_features = 4

    def test_supported_types_constant(self):
        self.assertEqual(SUPPORTED_TYPES, {"nf4", "fp4", "int8", "fp16", "bf16", "fp32"})

    def test_bitsandbytes_nf4(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features, bias=True)
        qlayer = self.quantizer.quantize_linear_layer(float_linear, "nf4")
        self.assertIsInstance(qlayer, bnb.nn.Linear4bit)
        self.assertEqual(qlayer.weight.shape, float_linear.weight.shape)

    def test_bitsandbytes_fp4(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        qlayer = self.quantizer.quantize_linear_layer(float_linear, "fp4")
        self.assertIsInstance(qlayer, bnb.nn.Linear4bit)
        self.assertEqual(qlayer.weight.shape, float_linear.weight.shape)

    def test_bitsandbytes_int8(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features, bias=True)
        qlayer = self.quantizer.quantize_linear_layer(float_linear, "int8")
        self.assertIsInstance(qlayer, bnb.nn.Linear8bitLt)

    def test_torch_fp16(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        qlayer = self.quantizer.quantize_linear_layer(float_linear, "fp16")

        self.assertIsInstance(qlayer, torch.nn.Linear)
        self.assertEqual(qlayer.weight.dtype, torch.float16)

        self.assertTrue(
            torch.allclose(
                qlayer.weight.float(),
                float_linear.weight.float(),
                atol=1e-3,
                rtol=1e-3
            )
        )

    def test_torch_bf16(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features, bias=True)
        qlayer = self.quantizer.quantize_linear_layer(float_linear, "bf16")
        self.assertIsInstance(qlayer, torch.nn.Linear)
        self.assertEqual(qlayer.weight.dtype, torch.bfloat16)

    def test_torch_fp32(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features)
        qlayer = self.quantizer.quantize_linear_layer(float_linear, "fp32")
        self.assertIsInstance(qlayer, torch.nn.Linear)
        self.assertEqual(qlayer.weight.dtype, torch.float32)
        # Check data is close
        self.assertTrue(torch.allclose(qlayer.weight, float_linear.weight.float(), atol=1e-7))

    def test_invalid_type(self):
        float_linear = torch.nn.Linear(self.in_features, self.out_features)
        with self.assertRaises(ValueError):
            self.quantizer.quantize_linear_layer(float_linear, "unknown")

    def test_mixed_quantizer(self):
        # Make a model with 3 layers
        model = MockModel(n_layers=3)
        # Each layer gets a different type
        schema = ["nf4", "int8", "bf16"]
        mq = MixedQuantizer(schema, quantizer=self.quantizer)
        quantized_model = mq.quantize_model(model, layer_attribute="model.layers")

        # layer0 => nf4 => bitsandbytes Linear4bit
        for subm in quantized_model.model.layers[0].children():
            self.assertIsInstance(subm, bnb.nn.Linear4bit)

        # layer1 => int8 => bitsandbytes Linear8bitLt
        for subm in quantized_model.model.layers[1].children():
            self.assertIsInstance(subm, bnb.nn.Linear8bitLt)

        # layer2 => bf16 => standard PyTorch linear in bfloat16
        for subm in quantized_model.model.layers[2].children():
            self.assertIsInstance(subm, torch.nn.Linear)
            self.assertEqual(subm.weight.dtype, torch.bfloat16)

    def check_numerical_similarity(self, original, quantized, dtype, tol=1e-3):
        original, quantized = original.to(torch.float32), quantized.to(torch.float32)
        self.assertTrue(torch.allclose(original, quantized, atol=tol), f"Numerical mismatch for dtype={dtype}")

    def test_quantize_supported_types(self):
        for qtype in SUPPORTED_TYPES:
            layer = torch.nn.Linear(self.in_features, self.out_features)
            quantized = self.quantizer.quantize_linear_layer(layer, qtype)
            self.assertIsNotNone(quantized.weight)

    def test_numerical_correctness_fp16(self):
        layer = torch.nn.Linear(self.in_features, self.out_features)
        quantized = self.quantizer.quantize_linear_layer(layer, "fp16")
        self.check_numerical_similarity(layer.weight, quantized.weight, dtype="fp16")

    def test_device_consistency(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        layer = torch.nn.Linear(self.in_features, self.out_features).to(device)
        quantized = self.quantizer.quantize_linear_layer(layer, "fp16")
        self.assertEqual(quantized.weight.device, device)

    def test_invalid_quant_type(self):
        layer = torch.nn.Linear(self.in_features, self.out_features)
        with self.assertRaises(ValueError):
            self.quantizer.quantize_linear_layer(layer, "invalid")

    def test_mixed_quantizer_schema_extension(self):
        model = MockModel(n_layers=5)
        schema = ["nf4", "int8"]
        mq = MixedQuantizer(schema, self.quantizer)
        quantized_model = mq.quantize_model(model, layer_attribute="model.layers")
        expected_types = [bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt,
                          bnb.nn.Linear4bit]

        for layer, expected in zip(quantized_model.model.layers, expected_types):
            self.assertIsInstance(layer.fc1, expected)
            self.assertIsInstance(layer.fc2, expected)

    def test_mixed_quantizer_nested_layers(self):
        # Make a model with nested layers
        model = NestedMockModel(n_layers=3)
        schema = ["nf4", "int8", "bf16"]
        mq = MixedQuantizer(schema, quantizer=self.quantizer)
        quantized_model = mq.quantize_model(model, layer_attribute="model.layers")

        # Check nested layer quantization
        self.assertIsInstance(quantized_model.model.layers[0].fc1[0], bnb.nn.Linear4bit)
        self.assertIsInstance(quantized_model.model.layers[0].fc1[1], bnb.nn.Linear4bit)

    def test_conv1d_to_linear_conversion(self):
        # Create a Conv1D layer
        conv1d_layer = Conv1D(self.in_features, self.out_features)
        conv1d_layer.weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        conv1d_layer.bias = torch.nn.Parameter(torch.randn(self.out_features))

        # Quantize the Conv1D layer
        quantized_layer = self.quantizer.quantize_linear_layer(conv1d_layer, "nf4")

        # Check that the quantized layer is an instance of Linear4bit
        self.assertIsInstance(quantized_layer, bnb.nn.Linear4bit)

        # Check that the weight shape is correct
        self.assertEqual(quantized_layer.weight.shape, (self.out_features, self.in_features))

    def test_quantize_conv1d_layer(self):
        # Create a Conv1D layer
        conv1d_layer = Conv1D(self.in_features, self.out_features)
        conv1d_layer.weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        conv1d_layer.bias = torch.nn.Parameter(torch.randn(self.out_features))

        # Quantize the Conv1D layer
        quantized_layer = self.quantizer.quantize_linear_layer(conv1d_layer, "int8")

        # Check that the quantized layer is an instance of Linear8bitLt
        self.assertIsInstance(quantized_layer, bnb.nn.Linear8bitLt)

        # Check that the weight shape is correct
        self.assertEqual(quantized_layer.weight.shape, (self.in_features, self.out_features))

    def test_mixed_quantizer_with_conv1d(self):
        # Create a fake GPT-2 model with Conv1D layers
        model = GPT2MockModel(n_layers=3)
        schema = ["nf4", "int8", "fp16"]
        mq = MixedQuantizer(schema, quantizer=self.quantizer)
        quantized_model = mq.quantize_model(model, layer_attribute="h")

        # Check that the layers are quantized correctly
        for layer in quantized_model.h:
            self.assertIsInstance(layer.c_attn, bnb.nn.Linear4bit)
            self.assertIsInstance(layer.c_proj, bnb.nn.Linear8bitLt)
            self.assertIsInstance(layer.c_fc, torch.nn.Linear)
            self.assertEqual(layer.c_fc.weight.dtype, torch.float16)

    def test_numerical_correctness_conv1d(self):
        # Create a Conv1D layer
        conv1d_layer = Conv1D(self.in_features, self.out_features)
        conv1d_layer.weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        conv1d_layer.bias = torch.nn.Parameter(torch.randn(self.out_features))

        # Quantize the Conv1D layer
        quantized_layer = self.quantizer.quantize_linear_layer(conv1d_layer, "fp16")

        # Generate random input
        input_tensor = torch.randn(1, self.in_features)

        # Compute outputs
        original_output = conv1d_layer(input_tensor)
        quantized_output = quantized_layer(input_tensor)

        # Check that the outputs are close
        self.assertTrue(torch.allclose(original_output, quantized_output, atol=1e-3))

if __name__ == "__main__":
    unittest.main()