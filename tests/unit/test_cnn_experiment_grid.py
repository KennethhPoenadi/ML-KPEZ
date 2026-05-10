from tubes2_ml.experiments.cnn_grid import generate_shared_conv_grid

def test_generate_shared_conv_grid_creates_16_experiments():
    configs = generate_shared_conv_grid()

    assert len(configs) == 16
    assert len({config.name for config in configs}) == 16

def test_generate_shared_conv_grid_covers_required_variations():
    configs = generate_shared_conv_grid()

    layer_counts = {len(config.conv_filters) for config in configs}
    filter_sets = {config.conv_filters for config in configs}
    kernel_sets = {config.kernel_sizes for config in configs}
    pooling_types = {config.pooling_type for config in configs}

    assert layer_counts == {1, 2}
    assert (16,) in filter_sets
    assert (32,) in filter_sets
    assert (16, 32) in filter_sets
    assert (32, 64) in filter_sets
    assert (3,) in kernel_sets
    assert (5,) in kernel_sets
    assert (3, 3) in kernel_sets
    assert (5, 5) in kernel_sets
    assert pooling_types == {"max", "average"}