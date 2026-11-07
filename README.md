# AI Class Labs

Coursework notebooks exploring linear regression, optimization, softmax regression, multilayer perceptrons, and image classification with PyTorch and TensorFlow/Keras. The repository contains both lab templates and worked experiments.

## Contents

| Path | Topics |
| --- | --- |
| `lab_practice/` | Linear equations, analytic regression, gradient descent, mini-batches, and MLPs |
| `lab_practices_Softmax_Regression_Implementation/` | Softmax regression |
| `lab_alexnet/` | AlexNet classification and feature-visualization lab template |
| `lab_vgg/` | Pet-expression preprocessing and VGG19/EfficientNet-B0 experiments |
| `YOLO_Object_Detection/` | YOLOv8 blood cell detection lab adapted to RBC-only detection, with NVIDIA GPU training and evaluation in the notebook |
| `pytorch_core_modules.ipynb` | PyTorch module exercises |

## Open a lab

Use Python 3 with Jupyter or VS Code's notebook support. A minimal Jupyter environment can be created from the repository root:

```sh
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then run:

```sh
python -m pip install jupyterlab
python -m jupyterlab
```

Choose the notebook and install the libraries used by its import cells in the same environment. Labs use different stacks, including NumPy, Matplotlib, pandas, scikit-learn, OpenCV, PyTorch/torchvision, and TensorFlow; there is no shared dependency lock or single command for all labs.

## Data and current limits

Read each notebook's instructions before running cells. The AlexNet template expects rock/paper/scissors downloads, and the VGG lab expects pet-expression images in `lab_vgg/pet_facial_expression_dataset/`. Only a metadata file is currently tracked inside that dataset folder, so supply the image class folders before running the VGG experiments.

Some cells are unfinished or depend on local data paths and helper definitions. Training may download pretrained weights, take substantial time, and write model or report files. Existing histories and reports are saved experiment artifacts.
