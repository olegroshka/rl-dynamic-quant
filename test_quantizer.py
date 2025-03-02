# test_quantizer.py

import unittest
import torch
import bitsandbytes as bnb
from quantizer import Quantizer, MixedQuantizer, SUPPORTED_TYPES

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

if __name__ == "__main__":
    unittest.main()
