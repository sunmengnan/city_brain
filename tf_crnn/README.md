# CRNN Tensorflow

原始论文: [An End-to-End Trainable Neural Network for Image-based
Sequence Recognition and Its Application to Scene Text Recognition](http://arxiv.org/abs/1507.05717)

原始代码: http://github.com/bgshih/crnn

## 环境准备
获得字符集：
```
cd data
git clone git@git.tianrang-inc.com:tianshi/ocr_chars.git
```

安装依赖:
```
pip3 install -r requirement.txt
```



## 训练数据
使用 [Text Renderer](http://git.tianrang-inc.com/tianshi/text_renderer) 生成训练集、验证集.
支持使用 TFRecord 、jpg 图片训练


## 训练
训练入口文件：train.py

Supported base cnn network and corresponding hyper parameters are stored in `./data/cfgs/` folder.

```shell
python3 train.py \
--tag=2018_11_26 \
--train_dir=path/to/your/training/images \
--val_dir=path/to/your/val/images \
--test_dir=path/to/your/test/images \
--chars_file=./data/chars/chn.txt \
--cfg_name=raw \
--gpu
```

## Inference
Download pretrained model from [here](https://pan.baidu.com/s/1Tt_WE6W4EIFE9NfYy7hbDw), extract it in `./output/checkpoint/default`, than run:

Images in `./data/demo` should have 32px height.

```shell
python3 infer.py \
--infer_dir=./data/demo \
--chars_file=./data/chars/chn.txt \
--infer_batch_size=1 \
--ckpt_dir=./output/checkpoint \
--result_dir=./output/result \
--tag=default
```

Output：
```bash
Batch [0/7] 2.772s accuracy: 1.000 (1/1), edit_distance: 0.0
Batch [1/7] 0.053s accuracy: 1.000 (1/1), edit_distance: 0.0
Batch [2/7] 0.054s accuracy: 1.000 (1/1), edit_distance: 0.0
Batch [3/7] 0.055s accuracy: 1.000 (1/1), edit_distance: 0.0
Batch [4/7] 0.054s accuracy: 1.000 (1/1), edit_distance: 0.0
Batch [5/7] 0.053s accuracy: 1.000 (1/1), edit_distance: 0.0
Batch [6/7] 0.056s accuracy: 1.000 (1/1), edit_distance: 0.0
Accuracy: 1.000 (7/7), Average edit distance: 0.000
Write result to ./output/result/default/infer/1.000.txt
```

