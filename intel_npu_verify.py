"""
Verifies installation of Intel NPU Drivers
for use with PyTorch
"""


def check_tensorflow():
    # Intel® Extension for TensorFlow* (CPU): python -c
    import intel_extension_for_tensorflow as itex

    print(itex.__version__)

    # Intel® Extension for TensorFlow* (GPU): python -c
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())


def check_pytorch():
    # Intel® Extension for PyTorch* (CPU): python -c
    import torch
    import intel_extension_for_pytorch as ipex

    print(torch.__version__)
    print(ipex.__version__)

    # Intel® Extension for PyTorch* (GPU): python -c
    [
        print(f"[{i}]: {torch.xpu.get_device_properties(i)}")
        for i in range(torch.xpu.device_count())
    ]


def check_sci_kit_learn():
    # Intel® Extension for Scikit-learn*: python -c
    from sklearnex import patch_sklearn

    patch_sklearn()


def check_xgboost():
    # Intel® Optimization for XGBoost*: python -c
    import xgboost as xgb

    print(xgb.__version__)


def check_neural_compressor():
    # Intel® Neural Compressor: python -c
    import neural_compressor as inc

    print(inc.__version__)


def check_modin():
    # Intel® Distribution of Modin*: python -c
    import modin

    print(modin.__version__)


def main():
    pytorch = True
    tensorflow = False
    xgboost = False
    scikit_learn = False
    modin = False
    neural_compressor = False

    if pytorch:
        check_pytorch()

    if tensorflow:
        check_tensorflow()

    if xgboost:
        check_xgboost()

    if scikit_learn:
        check_sci_kit_learn()

    if modin:
        check_modin()

    if neural_compressor:
        check_neural_compressor()


if __name__ == "__main__":
    main()
