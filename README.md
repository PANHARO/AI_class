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

## Lab summaries

- [Linear equations](lab_practice/Lab1_Mathematic_Linear_Equation.ipynb): Explores how weights and bias change the shape of a linear equation.
- [Linear regression and optimization](lab_practice/Lab1.2_Linear_Regression_Analytic_GD_MiniBatchSGD_Practices_v2.ipynb): Fits a line to sample data and explores the analytic solution, gradient descent, and mini-batch stochastic gradient descent. The [analytic solution lab](lab_practice/Lab1.2_Linear_Regression_Analytic_Solution_Practices.ipynb) focuses on finding the best-fitting parameters.
- [Delivery time optimization](lab_practice/AI_Lab_90min_Linear_Regression_Optimization.ipynb): Predicts food delivery time from distance while practicing linear regression optimization.
- [PyTorch multiple regression](lab_practice/lab3_multiple_linear_regression_pytorch_food_delivery_thermometer.ipynb): Predicts food delivery time from multiple order features using a PyTorch linear regression model.
- [Softmax regression](lab_practices_Softmax_Regression_Implementation/Softmax_Regression.ipynb): Classifies Fashion-MNIST clothing images into ten categories using a softmax model implemented from scratch.
- [Multilayer perceptron](lab_practice/new_lab_multilayer_perceptrons/Multilayer_perceptron_with_Generalization_Dropout_Practice.ipynb): Practices image classification with a neural network and investigates overfitting, validation, and dropout.
- [AlexNet](lab_alexnet/lab_practice.ipynb): Classifies rock, paper, and scissors images and examines feature maps learned by convolutional layers.
- [Pet expression classification](lab_vgg/binary_classification_train.ipynb): Trains and compares VGG19 and EfficientNet-B0 models on pet facial expression images; the [test notebook](lab_vgg/binary_classification_test.ipynb) evaluates models.
- [YOLO object detection](YOLO_Object_Detection/README.md): Trains and evaluates a YOLO model to locate red blood cells in images.
- [PyTorch core modules](pytorch_core_modules.ipynb): Practices neural network layers, losses, optimizers, and data loaders using small examples.
