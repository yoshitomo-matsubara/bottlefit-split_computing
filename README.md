# BottleFit for Split Computing

The official implementations of BottleFit for ILSVRC 2012 (ImageNet) dataset:
"BottleFit: Learning Compressed Representations in Deep Neural Networks for Effective and Efficient Split Computing," [IEEE WoWMoM '22](https://computing.ulster.ac.uk/WoWMoM2022/index.html)  
[[Paper](https://ieeexplore.ieee.org/document/9842809)] [[Preprint](https://arxiv.org/abs/2201.02693)]  

**More advanced work *"SC2 Benchmark: Supervised Compression for Split Computing"* and code** are available at https://github.com/yoshitomo-matsubara/sc2-benchmark
  
## Citations
```bibtex
@inproceedings{matsubara2022bottlefit,
  title={{BottleFit: Learning Compressed Representations in Deep Neural Networks for Effective and Efficient Split Computing}},
  author={Matsubara, Yoshitomo and Callegaro, Davide and Singh, Sameer and Levorato, Marco and Restuccia, Francesco},
  booktitle={2022 IEEE 23rd International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM)}, 
  pages={337-346},
  year={2022}
}
```

## Requirements
- Python >=3.6
- pipenv


## How to clone
```
git clone https://github.com/yoshitomo-matsubara/bottlefit-split_computing.git
cd bottlefit-split_computing/
pipenv install
```


## Download datasets
As the terms of use do not allow to distribute the URLs, you will have to create an account [here](http://image-net.org/download) to get the URLs, and replace `${TRAIN_DATASET_URL}` and `${VAL_DATASET_URL}` with them.
```
wget ${TRAIN_DATASET_URL} ./
wget ${VAL_DATASET_URL} ./
```

### ILSVRC 2012 (ImageNet) dataset
```
# Go to home directory
mkdir ~/dataset/ilsvrc2012/{train,val} -p
mv ILSVRC2012_img_train.tar ~/dataset/ilsvrc2012/train/
mv ILSVRC2012_img_val.tar ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/train/
tar -xvf ILSVRC2012_img_train.tar
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xf ../$f)
done
rm -r *.tar

wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/val/
tar -xvf ILSVRC2012_img_val.tar
sh valprep.sh
```

## Trained models

### Baseline methods
Our baseline results (Vanilla, KD, HND, and Autoencoder) and trained model weights are available at https://github.com/yoshitomo-matsubara/head-network-distillation

### Proposed methods
Create a directory and download the checkpoints of model weights ([densenet.zip](https://github.com/yoshitomo-matsubara/bottlefit-split_computing/releases/download/google_drive-to-github/densenet.zip), [resnet.zip](https://github.com/yoshitomo-matsubara/bottlefit-split_computing/releases/download/google_drive-to-github/resnet.zip), [resnet-other_vers.zip](https://github.com/yoshitomo-matsubara/bottlefit-split_computing/releases/download/google_drive-to-github/resnet-other_vers.zip), [resnet-other_methods.zip](https://github.com/yoshitomo-matsubara/bottlefit-split_computing/releases/download/google_drive-to-github/resnet-other_methods.zip))

```shell
mkdir -p resource/ckpt
```

Unzip the downloaded zip files under `./resource/ckpt/`, then there will be `./resource/ckpt/image_classification/`.

## Test trained models
Taking 3ch-bottleneck-injected models as examples
### GHND-KD
```shell
pipenv run python image_classification.py --config config/ghnd_kd/custom_densenet169_from_densenet169-3ch -test_only
pipenv run python image_classification.py --config config/ghnd_kd/custom_densenet201_from_densenet201-3ch.yaml -test_only
pipenv run python image_classification.py --config config/ghnd_kd/custom_resnet152_from_resnet152-3ch.yaml -test_only
```

### GHND-KD (FE)
```shell
pipenv run python image_classification.py --config config/ghnd_kd-finetune/custom_densenet169_from_densenet169-3ch -test_only
pipenv run python image_classification.py --config config/ghnd_kd-finetune/custom_densenet201_from_densenet201-3ch.yaml -test_only
pipenv run python image_classification.py --config config/ghnd_kd-finetune/custom_resnet152_from_resnet152-3ch.yaml -test_only
```

### GHND-FT
```shell
pipenv run python image_classification.py --config config/ghnd_vanilla/custom_densenet169_from_densenet169-3ch.yaml -test_only
pipenv run python image_classification.py --config config/ghnd_vanilla/custom_densenet201_from_densenet201-3ch.yaml -test_only
pipenv run python image_classification.py --config config/ghnd_vanilla/custom_resnet152_from_resnet152-3ch.yaml -test_only
```

### GHND-FT (FE)
```shell
pipenv run python image_classification.py --config config/ghnd_vanilla-finetune/custom_densenet169_from_densenet169-3ch.yaml -test_only
pipenv run python image_classification.py --config config/ghnd_vanilla-finetune/custom_densenet201_from_densenet201-3ch.yaml -test_only
pipenv run python image_classification.py --config config/ghnd_vanilla-finetune/custom_resnet152_from_resnet152-3ch.yaml -test_only
```

## Train models
If you would like to train models, you should exclude `-test_only` from the above commands, and set new file paths for student model in the yaml files.  
To enable the distributed training mode, you should use `pipenv run python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env image_classification.py  ... --world_size ${NUM_GPUS}`
