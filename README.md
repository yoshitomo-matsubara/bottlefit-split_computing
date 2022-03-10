# BottleFit for Split Computing

The official implementations of BottleFit for ILSVRC 2012 (ImageNet) dataset:
"BottleFit: Learning Compressed Representations in Deep Neural Networks for Effective and Efficient Split Computing," [IEEE WoWMoM '22](https://computing.ulster.ac.uk/WoWMoM2022/index.html)  
[[Preprint](https://arxiv.org/abs/2201.02693)]  
  
## Citations
```bibtex
@article{matsubara2022bottlefit,
  title={BottleFit: Learning Compressed Representations in Deep Neural Networks for Effective and Efficient Split Computing},
  author={Matsubara, Yoshitomo and Callegaro, Davide and Singh, Sameer and Levorato, Marco and Restuccia, Francesco},
  journal={arXiv preprint arXiv:2201.02693},
  year={2022}
}
```

## Requirements
- Python >=3.6
- pipenv
- [myutils](https://github.com/yoshitomo-matsubara/myutils)


## How to clone
```
git clone https://github.com/yoshitomo-matsubara/bottlefit-split_computing.git
cd bottlefit-split_computing/
git submodule init
git submodule update --recursive --remote
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
Create a directory and download the checkpoints of model weights at https://drive.google.com/file/d/1uFK-z_vincaujhm_LnKo_rno6G3B_FkU/view?usp=sharing

```shell
mkdir -p resource/ckpt
```

Unzip the downloaded zip files under `./resource/ckpt/`, then there will be `./resource/ckpt/image_classification/`.

## Test trained models

## Train models
If you would like to train models, you should exclude `-test_only` from the above commands, and set new file paths for student model in the yaml files.  
To enable the distributed training mode, you should use `pipenv run python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env image_classification.py  ... --world_size ${NUM_GPUS}`
