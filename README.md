# SiamMGT

## 1. Environment setup
This code has been tested on Ubuntu 20.04, Python 3.8, Pytorch 1.12.0, CUDA 11.3.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test

<table width="80%" border="1" cellpadding="8" style="border-collapse:collapse; margin:0 auto;">
    <tr>
        <td colspan="2" align="center"><strong>Dataset</strong></td>
        <td align="center"><strong>SiamMGT</strong></td>
    </tr>
    <tr>
        <td rowspan="3" align="center" valign="middle">GOT10k</td>
        <td align="center">AO</td>
        <td align="center">65.9</td>
    </tr>
    <tr>
        <td align="center">SR0.5</td>
        <td align="center">76.7</td>
    </tr>
    <tr>
        <td align="center">SR0.75</td>
        <td align="center">56.1</td>
    </tr>
    <tr>
        <td rowspan="3" align="center" valign="middle">LaSOT</td>
        <td align="center">Success</td>
        <td align="center">58.0</td>
    </tr>
    <tr>
        <td align="center">Norm precision</td>
        <td align="center">0</td>
    </tr>
    <tr>
        <td align="center">Precision</td>
        <td align="center">0</td>
    </tr>
    <tr>
        <td rowspan="3" align="center" valign="middle">TrackingNet</td>
        <td align="center">Success</td>
        <td align="center">77.3</td>
    </tr>
    <tr>
        <td align="center">Norm precision</td>
        <td align="center">82.6</td>
    </tr>
    <tr>
        <td align="center">Precision</td>
        <td align="center">73.1</td>
    </tr>
    <tr>
        <td rowspan="2" align="center" valign="middle">OTB100</td>
        <td align="center">Success</td>
        <td align="center">71.0</td>
    </tr>
    <tr>
        <td align="center">Precision</td>
        <td align="center">91.7</td>
    </tr>
    <tr>
        <td rowspan="2" align="center" valign="middle">UAV123</td>
        <td align="center">Success</td>
        <td align="center">65.5</td>
    </tr>
    <tr>
        <td align="center">Precision</td>
        <td align="center">84.8</td>
    </tr>
</table>


### Prepare testing datasets
Download testing datasets and put them into `test_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

### Test the tracker
```bash 
python testTracker.py \    
        --config ../experiments/siammgt/config.yaml \ 
	--dataset OTB100 \                                 # dataset_name: GOT-10k, LaSOT, TrackingNet, OTB100, UAV123
	--snapshot snapshot/model.pth              # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train

### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)
* [LaSOT](https://cis.temple.edu/lasot/)
* [TrackingNet](https://tracking-net.org/#downloads)

**Note:** `training_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

### Download pretrained backbones
Download pretrained backbones from [link](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) and put them into `pretrained_models` directory.

### Train a model
To train the SiamMGT model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
	--cfg ../experiments/siammgt/config.yaml 
```

## 4. Evaluation

We provide tracking results for comparison: 
- SiamMGT/results: [BaiduYun](https://pan.baidu.com/s/1HBE0Kn2ietvQT7NLExAQoA) (extract code: 8nox) 


If you want to evaluate the tracker on OTB100, UAV123 and LaSOT, please put those results into `results` directory and then run `eval.py` . 
Evaluate GOT-10k on [Server](http://got-10k.aitestunion.com/). Evaluate TrackingNet on [Server](https://tracking-net.org/).  

```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset OTB100                  \ # dataset_name
	--tracker_prefix 'model.pth'   # tracker_name
```

## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot) and [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT). We would like to express our sincere thanks to the contributors.

## 6. Cite
If you use SiamMGT in your work please cite our papers:


