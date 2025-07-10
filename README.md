# Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline
<p align="center">Zhihao Zhang<sup>1</sup>, Chunyu Lin<sup>1 *</sup>, Lang Nie<sup>1</sup>, Jiyuan Wang<sup>1</sup>, Yao Zhao<sup>1</sup></p>
<p align="center"><sup>1</sup>Beijing Jiaotong University</p>

## Dataset (CRPS-D)
We build a large-scale parking slot detection dataset (named CRPS-D), which includes various lighting distributions, diverse weather conditions, and challenging parking slot variants.

Now, the dataset can be downloaded in in [Google Drive](https://drive.google.com/file/d/10Lm7RdoMliTVDnYX9lnM_Z3qXytXDWbG).
The annotation tool and corresponding instructions can be downloaded [here](https://drive.google.com/file/d/1muVTCgz8Tg6dSIZy7Ql4zcT7r8ZWsfxq).


## Baseline (SS-PSD)

Additionally, we develop a semi-supervised baseline for parking slot detection, termed SS-PSD, to further improve performance by exploiting unlabeled data.
This is the implementation of **SS-PSD** using PyTorch.

## Requirements

* PyTorch
* CUDA (optional)
* Other requirements  
    `pip install -r requirements.txt`


## Inference

#### Pretrained model
The pretrained model we trained on the CRPS-D dataset can be available at [Google Drive](https://drive.google.com/file/d/11gJcAly9MXBbe02PQ5g4Uuhi8AUnza1S/view?usp=drive_link).

    ```(shell)
    python inference.py --mode image --detector_weights $DETECTOR_WEIGHTS --inference_slot
    ```

    Argument `DETECTOR_WEIGHTS` is the trained weights of detector.  


## Train

```(shell)
python train.py
```

The folder structure of the training data can be found in the yaml/data_root.yaml file.


## Evaluate

* Evaluate directional marking-point detection

    ```(shell)
    python evaluate.py --detector_weights $DETECTOR_WEIGHTS
    ```
    ```(shell)
    python evaluate.py --eval_all True
    ```

    Argument `eval_all` determine whether to run all weight files (True or False).  


* Evaluate parking-slot detection

    ```(shell)
    python ps_evaluate.py  --detector_weights $DETECTOR_WEIGHTS
    ```
    ```(shell)
    python ps_evaluate.py --eval_all True
    ```

    Argument `eval_all` determine whether to run all weight files (True or False).  


