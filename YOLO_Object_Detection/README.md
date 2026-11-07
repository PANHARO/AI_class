# Object Detection Lab 1

Student practice: adapt the supplied blood cell dataset to detect **RBC only**.
The notebook covers GPU checking, dataset preparation, training,
prediction comparisons, IoU and mAP evaluation in executable code cells.

## Environment (Windows PowerShell)

Use your existing **Python 3.14** installation. It already has PyTorch
`2.12.1+cu130`, torchvision `0.27.1+cu130`, Ultralytics `8.4.90`, and Jupyter kernel
support. A CUDA matrix calculation was checked on the RTX 5060 Ti. No environment
installation is needed on this PC.

From the repository root:

```powershell
cd YOLO_Object_Detection
```

If you need to reinstall dependencies later, run
`py -3.14 -m pip install -r requirements.txt`. The requirements use CUDA 13.0
builds. A separate CUDA Toolkit installation is not needed for prebuilt wheels.

Open `yolo_detect_blood_cell_fixed.ipynb` in VS Code and select
**Python 3.14** as its kernel, then click **Run All**. The cells prepare the data,
train the model, show predictions and evaluate the best checkpoint in order.
Running all cells again starts a new 5-epoch training run.

## Dataset

The supplied data lives in `dataset/blood_cell_dataset`:

| Split | Images | RBC boxes |
| --- | ---: | ---: |
| Train | 765 | 8,814 |
| Validation | 73 | 819 |
| Test | 36 | 398 |

The source class order is `Platelets, RBC, WBC`. The notebook copies all images
into `dataset/rbc_only`, retains RBC annotations and remaps class 1 to class 0.
Original labels and split assignments are preserved. It creates `data_rbc.yaml`
for the custom task and corrects `data.yaml` for the original three-class task.
Run the preparation cells again after moving the project to regenerate paths.

## Train in the notebook

Section 6 contains the executable `model.train(...)` cell.

This follows the lab's model, epoch count and image size. GPU 0 is the NVIDIA
RTX 5060 Ti. Batch size 4 is a conservative starting point for 16 GB VRAM;
close other GPU-heavy applications or reduce it to 2 if needed.
`workers=0` avoids Windows multiprocessing setup. Pretrained YOLOv8x weights
download on the first run. Training evaluates the validation split each epoch.

After training, notebook sections 7-9 inspect the curves and compare
ground-truth boxes with predictions on test images.

## Test evaluation in the notebook

Section 11 contains the executable `test_model.val(...)` cell. It evaluates
the best checkpoint on the held-out test split and saves evaluation plots.

Record precision, recall, mAP50 and mAP50-95 from the cell output in the final
notebook observations. No scores are claimed before you run the experiment.

For an optional original three-class baseline, change the training cell to use
`data=str(base_dir / "data.yaml")` and `name="blood_cells_yolov8x"`.

## References

- Assignment: `../Object Detection_Lab1.pdf`
- [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)
- [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/)
