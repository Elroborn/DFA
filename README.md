<!--
 * @Descripttion: 
 * @Author: coderwangson
 * @Date: 2020-12-02 10:00:13
 * @FilePath: /py35/Ablation_Experiment/README.md
 * @LastEditTime: 2020-12-02 10:13:06
-->
# Usage
## Main Dependencies 

```
pytorch
opencv
numpy
```

## Prepare Dataset 

We need to prepare 'file_list.txt', the data format inside is as follows:

```
/data/oulu/spatial/Train_files/1_1_01_1_avi/frame4.jpg /data/oulu/depth/Train_files/1_1_01_1_avi/frame4.jpg 1
```
the `/data/oulu/spatial/Train_files/1_1_01_1_avi/frame4.jpg` is the location of the RGB picture, and `/data/oulu/depth/Train_files/1_1_01_1_avi/frame4.jpg` is the position corresponding to the Depth picture. `1` is label.

## Train and Test 

### Train  

You can modify the configuration according to your needs by modifying `opt.py` . And run :
```
python main.py
```
to start a train.  

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