# Usage
## Main Dependencies 

```
python>=3.5
pytorch>=1.0.0
opencv
numpy
```
The hardware environment is NVIDIA GTX 1080Ti.
## Prepare Dataset 

We need to prepare 'file_list.txt', the data format inside is as follows:

```
/data/oulu/spatial/Train_files/1_1_01_1_avi/frame4.jpg /data/oulu/depth/Train_files/1_1_01_1_avi/frame4.jpg 1
```
the `/data/oulu/spatial/Train_files/1_1_01_1_avi/frame4.jpg` is the location of the RGB picture, and `/data/oulu/depth/Train_files/1_1_01_1_avi/frame4.jpg` is the position corresponding to the Depth picture. `1` is label.

## Train and Test 

### Train  

You can modify the configuration according to your needs by modifying `options.py` . And run :
```
python main.py
```
to start a train.  

The directory of the trained model is determined by `--checkpoints_dir` and `--name` in `options.py`.

The `--model` in `options.py` represents three different models, you can get specific details from the ablation experiment of the paper.

If there is a `CUDA out of memory` problem, you can modify `gpu_ids` or `batch_size` in `options.py`.

### Test

We provide three models in the `checkpoints`. You can run:
```
python test.py  --name model1
```

```
python test.py  --name model2
```

```
python test.py  --name model2
```
to test different models.