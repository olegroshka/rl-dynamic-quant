import visualization as vis

# vis.plot_single_layer_types(['fp32', 'fp16', 'int8', 'nf4', 'fp16', 'int8', 'nf4', 'int8', 'nf4', 'fp16', 'fp32', 'fp32'], 'Layer', 'Precision', 'Single Layer Quantization Configuration', 'layer_config_plot.png')

configurations = {
    "gpt2-small-boolq-a": ['fp16', 'int8', 'fp16', 'nf4', 'fp16', 'int8', 'fp16', 'int8', 'fp4', 'nf4', 'int8', 'fp4'],
    "gpt2-small-boolq-b": ['fp16', 'nf4', 'nf4', 'fp16', 'nf4', 'nf4', 'nf4', 'int8', 'int8', 'nf4', 'nf4', 'fp4'],

    "gpt2-small-piqa-a": ['fp16', 'int8', 'fp16', 'nf4', 'fp16', 'int8', 'fp16', 'int8', 'fp4', 'nf4', 'int8', 'fp4'],
    "gpt2-small-piqa-b": ['fp16', 'nf4', 'nf4', 'fp16', 'nf4', 'nf4', 'nf4', 'int8', 'int8', 'nf4', 'nf4', 'fp4'],

    "gpt2-medium-boolq-a": ['fp4', 'nf4', 'nf4', 'int8', 'nf4', 'nf4', 'fp16', 'fp4', 'fp4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'int8', 'nf4', 'nf4', 'fp4', 'nf4', 'fp4', 'int8'],
    "gpt2-medium-boolq-b": ['fp4', 'fp4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'int8', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'fp4', 'int8'],
    "gpt2-medium-boolq-c": ['fp4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp16'],

    "gpt2-medium-piqa-a":  ['fp4', 'nf4', 'nf4', 'int8', 'nf4', 'nf4', 'fp16', 'fp4', 'fp4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'int8', 'nf4', 'nf4', 'fp4', 'nf4', 'fp4', 'int8'],
    "gpt2-medium-piqa-b": ['fp4', 'fp4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'int8', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'fp4', 'int8'],
    "gpt2-medium-piqa-c": ['fp4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'fp16']
}

for name, conf in configurations.items():
    vis.plot_single_layer_types(
        conf, 'Layer', 'Precision', 'Single Layer Quantization Configuration', f'{name}.png'
    )
