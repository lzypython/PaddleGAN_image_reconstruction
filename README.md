# 基于PaddleGAN完成遥感图像超分辨率的迁移学习

## 一、项目背景
- **遥感影像超分辨率**重建的意义：遥感影像的分辨率对应用来说十分重要，无论是变化检测、目标检测还是地物分类等领域，高质量的遥感影像对于下游任务的结果有比较大的影响。
- **迁移学习**的意义：是一种机器学习方法，是把一个领域（即源领域）的知识，迁移到另外一个领域（即目标领域），使得目标领域能够取得更好的学习效果。可以归纳为三点：
    - 更高的起点：在微调之前，模型的初始性能更高；
    - 更高的斜率：训练过程中，模型提升的速率更快；
    - 更高的渐进：训练结束后，得到的模型收敛更好。
- **本项目目的**：PaddleGAN的[图像超分辨率的模型库](https://github.com/PaddlePaddle/PaddleGAN/edit/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)比较丰富，而且都已经在DIV2K数据集上训练好，权重可自行下载，不需要自己复现代码。但是由于本项目的目标数据是遥感影像，所以直接使用该权重**不能达到最优效果**。如果使用PaddleGAN重新从头训练，**消耗的算力太多了**，你说是吧？因此想到使用训练好的超分模型权重进行迁移学习，减少训练所用时间的同时达到令人满意的效果。
- 看看放大四倍的效果：

| 低分辨率 | 超分辨率 | 高分辨率 |
|---|---|---|
|![](https://ai-studio-static-online.cdn.bcebos.com/19a9c0a7d9944d64bc646f2b12ff9ab8d13b1ba723b540ee9e8eecff8b537da0)| ![](https://ai-studio-static-online.cdn.bcebos.com/ce099aedd8c1423384e4880616310c45421a0eb0ec3843cf9eee87dadcb24ea1)| ![](https://ai-studio-static-online.cdn.bcebos.com/17d262b729344bd398fd1449d237b5cb5be77be5f5ea43d2979e650043487812)|




## 二、准备工作
- 在进行训练之前，首先要克隆PaddleGAN，然后下载PaddleGAN提供的模型权重，再要准备数据，然后安装环境依赖。

- 项目所用数据集地址：[RS_SR_Data](https://aistudio.baidu.com/aistudio/datasetdetail/129011) ，简介有介绍该数据集如何产生，这里不做重复
- 从高分辨率影像生成对应低分辨率影像的代码放入work文件夹中，名称为：Prepare_TestData_HR_LR.m



```python
# 从gitee克隆PaddleGAN的仓库
# clone好后便不用执行
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git #从码云上clone
```

    正克隆到 'PaddleGAN'...
    remote: Enumerating objects: 5600, done.[K
    remote: Counting objects: 100% (1656/1656), done.[K
    remote: Compressing objects: 100% (768/768), done.[K
    接收对象中:  24% (1353/5600), 7.82 MiB | 281.00 KiB/s     



- PaddleGAN中的超分辨率模型在以下数据集的 RGB 通道上进行评估，并在评估之前裁剪每个边界的尺度像素。
度量指标为 PSNR / SSIM。
- 可以看到模型库中指标最高的模型是DRN，论文为CVPR2020的[Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution](https://arxiv.org/pdf/2003.07018.pdf)，故选用该模型的参数做为迁移学习的初始值


| 模型 | Set5 | Set14 | DIV2K |
|---|---|---|---|
| realsr_df2k  | 28.4385 / 0.8106 | 24.7424 / 0.6678 | 26.7306 / 0.7512 |
| realsr_dped  | 20.2421 / 0.6158 | 19.3775 / 0.5259 | 20.5976 / 0.6051 |
| realsr_merge  | 24.8315 / 0.7030 | 23.0393 / 0.5986 | 24.8510 / 0.6856 |
| lesrcnn_x4  | 31.9476 / 0.8909 | 28.4110 / 0.7770 | 30.231 / 0.8326 |
| esrgan_psnr_x4  | 32.5512 / 0.8991 | 28.8114 / 0.7871 | 30.7565 / 0.8449 |
| esrgan_x4  | 28.7647 / 0.8187 | 25.0065 / 0.6762 | 26.9013 / 0.7542 |
| pan_x4  | 30.4574 / 0.8643 | 26.7204 / 0.7434 | 28.9187 / 0.8176 |
| drns_x4  | 32.6684 / 0.8999 | 28.9037 / 0.7885 | - |


```python
# 获取PaddleGAN训练好的DRN模型权重，并保存到work/pretrain文件夹中，若已存在则不用运行下列命令
# 事先已经准备好，可不用执行
!wget https://paddlegan.bj.bcebos.com/models/DRNSx4.pdparams 
!mkdir work/pretrain
!mv DRNSx4.pdparams work/pretrain/
```

    --2022-03-21 10:57:43--  https://paddlegan.bj.bcebos.com/models/DRNSx4.pdparams
    Resolving paddlegan.bj.bcebos.com (paddlegan.bj.bcebos.com)... 182.61.200.195, 182.61.200.229, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddlegan.bj.bcebos.com (paddlegan.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19266672 (18M) [application/octet-stream]
    Saving to: ‘DRNSx4.pdparams’
    
    DRNSx4.pdparams     100%[===================>]  18.37M  15.0MB/s    in 1.2s    
    
    2022-03-21 10:57:44 (15.0 MB/s) - ‘DRNSx4.pdparams’ saved [19266672/19266672]
    
    mkdir: cannot create directory ‘work/pretrain’: File exists



```python
# 解压数据集到指定文件夹中，大概一分钟
# 执行过一次第二次不用执行
!unzip -oq data/data129011/RSdata_for_SR.zip -d PaddleGAN/data/
```


```python
!pwd
```

    /home/aistudio



```python
# 安装依赖
%cd /home/aistudio/PaddleGAN
!pip install -r requirements.txt 
```

    /home/aistudio/PaddleGAN
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (4.36.1)
    Requirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (5.1.2)
    Collecting scikit-image>=0.14.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2d/ba/63ce953b7d593bd493e80be158f2d9f82936582380aee0998315510633aa/scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.5/13.5 MB[0m [31m803.2 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: scipy>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (1.6.3)
    Requirement already satisfied: opencv-python<=4.6.0.66 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (4.1.1.26)
    Collecting imageio==2.9.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6e/57/5d899fae74c1752f52869b613a8210a2480e1a69688e65df6cb26117d45d/imageio-2.9.0-py3-none-any.whl (3.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.3/3.3 MB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 7)) (0.3.0)
    Collecting librosa==0.8.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/54/19/a0e2bdc94bc0d1555e4f9bc4099a0751da83fa6e1e6157ec005564f8a98a/librosa-0.8.1-py3-none-any.whl (203 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m203.8/203.8 kB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: numba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 9)) (0.48.0)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 10)) (1.9)
    Collecting munch
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cc/ab/85d8da5c9a45e072301beb37ad7f833cd344e04c817d97e0cc75681d248f/munch-2.5.0-py2.py3-none-any.whl (10 kB)
    Collecting natsort
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5f/e1/70d203ba3ae5476f25fb8a2015d6a7ff156a4ce4795e36955c144ea5a826/natsort-8.3.1-py3-none-any.whl (38 kB)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 13)) (2.2.3)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio==2.9.0->-r requirements.txt (line 6)) (1.20.3)
    Requirement already satisfied: pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio==2.9.0->-r requirements.txt (line 6)) (8.2.0)
    Collecting pooch>=1.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/84/8c/4da580db7fb4cfce8f5ed78e7d2aa542e6f201edd69d3d8a96917a8ff63c/pooch-1.7.0-py3-none-any.whl (60 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.9/60.9 kB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging>=20.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (21.3)
    Requirement already satisfied: decorator>=3.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (4.4.2)
    Requirement already satisfied: resampy>=0.2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.2.2)
    Requirement already satisfied: joblib>=0.14 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.14.1)
    Requirement already satisfied: audioread>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (2.1.8)
    Requirement already satisfied: soundfile>=0.10.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.10.3.post1)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.24.2)
    Collecting PyWavelets>=1.1.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ae/56/4441877073d8a5266dbf7b04c7f3dc66f1149c8efb9323e0ef987a9bb1ce/PyWavelets-1.3.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.4/6.4 MB[0m [31m1.1 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m0m
    [?25hCollecting tifffile>=2019.7.26
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d8/38/85ae5ed77598ca90558c17a2f79ddaba33173b31cf8d8f545d34d9134f0d/tifffile-2021.11.2-py3-none-any.whl (178 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m178.9/178.9 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: networkx>=2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.4)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba->-r requirements.txt (line 9)) (56.2.0)
    Requirement already satisfied: llvmlite<0.32.0,>=0.31.0dev0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba->-r requirements.txt (line 9)) (0.31.0)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from munch->-r requirements.txt (line 11)) (1.16.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->-r requirements.txt (line 13)) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->-r requirements.txt (line 13)) (3.0.9)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->-r requirements.txt (line 13)) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->-r requirements.txt (line 13)) (2.8.2)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->-r requirements.txt (line 13)) (2019.3)
    Requirement already satisfied: requests>=2.19.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (2.22.0)
    Collecting platformdirs>=2.5.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/89/7e/c6ff9ddcf93b9b36c90d88111c4db354afab7f9a58c7ac3257fa717f1268/platformdirs-3.5.1-py3-none-any.whl (15 kB)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->-r requirements.txt (line 8)) (2.1.0)
    Requirement already satisfied: cffi>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from soundfile>=0.10.2->librosa==0.8.1->-r requirements.txt (line 8)) (1.15.1)
    Requirement already satisfied: pycparser in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->-r requirements.txt (line 8)) (2.21)
    Collecting typing-extensions>=4.5
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/31/25/5abcd82372d3d4a3932e1fa8c3dbf9efac10cc7c0d16e78467460571b404/typing_extensions-4.5.0-py3-none-any.whl (27 kB)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (2019.9.11)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (1.25.6)
    Installing collected packages: typing-extensions, tifffile, PyWavelets, natsort, munch, imageio, scikit-image, platformdirs, pooch, librosa
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.3.0
        Uninstalling typing_extensions-4.3.0:
          Successfully uninstalled typing_extensions-4.3.0
      Attempting uninstall: imageio
        Found existing installation: imageio 2.6.1
        Uninstalling imageio-2.6.1:
          Successfully uninstalled imageio-2.6.1
      Attempting uninstall: librosa
        Found existing installation: librosa 0.7.2
        Uninstalling librosa-0.7.2:
          Successfully uninstalled librosa-0.7.2
    Successfully installed PyWavelets-1.3.0 imageio-2.9.0 librosa-0.8.1 munch-2.5.0 natsort-8.3.1 platformdirs-3.5.1 pooch-1.7.0 scikit-image-0.19.3 tifffile-2021.11.2 typing-extensions-4.5.0
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip available: [0m[31;49m22.1.2[0m[39;49m -> [0m[32;49m23.1.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


## 三、迁移学习与训练
- **迁移学习**：遥感影像超分辨率的迁移学习其实就是加载预训练模型微调，所以只需要设置好训练的配置文件，运行时使用命令load一下预训练模型即可
- 下面介绍配置文件drn_psnr_x4_rssr.yaml，其如下所示。其中：
	- total_iters是训练的迭代次数
   - output_dir为训练输出的文件夹名称
   - gt_folder为高分辨率影像所在文件夹，lq_folder为低分辨率所在文件夹，图片名要相互对应
   - scale为影像超分辨率的倍数，DRN只能选择2或者4，本项目选择4
   - 文件的最后snapshot_config: interval: 1000指的是每迭代1000次保存一次权重
 - 由于是迁移学习，所以不用训练太久，设置为20000，大概**一个半小时**就能达到好的效果

```
total_iters: 20000
output_dir: output_dir
# tensor range for function tensor2img
min_max:
  (0., 255.)

model:
  name: DRN
  generator:
    name: DRNGenerator
    scale: (2, 4)
    n_blocks: 30
    n_feats: 16
    n_colors: 3
    rgb_range: 255
    negval: 0.2
  pixel_criterion:
    name: L1Loss

dataset:
  train:
    name: SRDataset
    gt_folder: data/RSdata_for_SR/trian_HR
    lq_folder: data/RSdata_for_SR/train_LR/x4
    num_workers: 4
    batch_size: 8
    scale: 4
    preprocess:
      - name: LoadImageFromFile
        key: lq
      - name: LoadImageFromFile
        key: gt
      - name: Transforms
        input_keys: [lq, gt]
        output_keys: [lq, lqx2, gt]
        pipeline:
          - name: SRPairedRandomCrop
            gt_patch_size: 192
            scale: 4
            scale_list: True
            keys: [image, image]
          - name: PairedRandomHorizontalFlip
            keys: [image, image, image]
          - name: PairedRandomVerticalFlip
            keys: [image, image, image]
          - name: PairedRandomTransposeHW
            keys: [image, image, image]
          - name: Transpose
            keys: [image, image, image]
          - name: Normalize
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
            keys: [image, image, image]
  test:
    name: SRDataset
    gt_folder: data/RSdata_for_SR/test_HR
    lq_folder: data/RSdata_for_SR/test_LR/x4
    scale: 4
    preprocess:
      - name: LoadImageFromFile
        key: lq
      - name: LoadImageFromFile
        key: gt
      - name: Transforms
        input_keys: [lq, gt]
        pipeline:
          - name: Transpose
            keys: [image, image]
          - name: Normalize
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
            keys: [image, image]

lr_scheduler:
  name: CosineAnnealingRestartLR
  learning_rate: 0.0001
  periods: [1000000]
  restart_weights: [1]
  eta_min: !!float 1e-7

optimizer:
  optimG:
    name: Adam
    net_names:
      - generator
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.999
  optimD:
    name: Adam
    net_names:
      - dual_model_0
      - dual_model_1
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.999

validate:
  interval: 1000
  save_img: True

  metrics:
    psnr: # metric name, can be arbitrary
      name: PSNR
      crop_border: 4
      test_y_channel: True
    ssim:
      name: SSIM
      crop_border: 4
      test_y_channel: True

log_config:
  interval: 10
  visiual_interval: 500

snapshot_config:
  interval: 1000
```

- .yaml文件已放入work文件夹中，运行以下命令即可加载预训练模型开始训练


```python
!python -u tools/main.py --config-file ../work/drn_psnr_x4_rssr.yaml --load ../work/pretrain/DRNSx4.pdparams
```

    [05/29 18:26:09] ppgan.engine.trainer INFO: Iter: 13100/20000 lr: 9.996e-05 loss_promary: 8.114 loss_dual: 1.006 loss_total: 9.120 batch_cost: 0.19771 sec reader_cost: 0.00029 sec ips: 40.46246 images/s eta: 0:22:44
    [05/29 18:26:11] ppgan.engine.trainer INFO: Iter: 13110/20000 lr: 9.996e-05 loss_promary: 7.907 loss_dual: 0.949 loss_total: 8.856 batch_cost: 0.20322 sec reader_cost: 0.00029 sec ips: 39.36712 images/s eta: 0:23:20
    [05/29 18:26:13] ppgan.engine.trainer INFO: Iter: 13120/20000 lr: 9.996e-05 loss_promary: 9.708 loss_dual: 1.156 loss_total: 10.864 batch_cost: 0.20787 sec reader_cost: 0.00029 sec ips: 38.48474 images/s eta: 0:23:50
    [05/29 18:26:15] ppgan.engine.trainer INFO: Iter: 13130/20000 lr: 9.996e-05 loss_promary: 8.144 loss_dual: 0.938 loss_total: 9.082 batch_cost: 0.20754 sec reader_cost: 0.00030 sec ips: 38.54682 images/s eta: 0:23:45
    [05/29 18:26:17] ppgan.engine.trainer INFO: Iter: 13140/20000 lr: 9.996e-05 loss_promary: 8.536 loss_dual: 1.009 loss_total: 9.545 batch_cost: 0.19794 sec reader_cost: 0.00028 sec ips: 40.41711 images/s eta: 0:22:37
    [05/29 18:26:19] ppgan.engine.trainer INFO: Iter: 13150/20000 lr: 9.996e-05 loss_promary: 8.673 loss_dual: 1.008 loss_total: 9.681 batch_cost: 0.20917 sec reader_cost: 0.00030 sec ips: 38.24591 images/s eta: 0:23:52
    [05/29 18:26:21] ppgan.engine.trainer INFO: Iter: 13160/20000 lr: 9.996e-05 loss_promary: 8.257 loss_dual: 1.067 loss_total: 9.324 batch_cost: 0.21111 sec reader_cost: 0.00030 sec ips: 37.89580 images/s eta: 0:24:03
    [05/29 18:26:23] ppgan.engine.trainer INFO: Iter: 13170/20000 lr: 9.996e-05 loss_promary: 9.440 loss_dual: 1.060 loss_total: 10.500 batch_cost: 0.20126 sec reader_cost: 0.00030 sec ips: 39.74935 images/s eta: 0:22:54
    [05/29 18:26:25] ppgan.engine.trainer INFO: Iter: 13180/20000 lr: 9.996e-05 loss_promary: 8.957 loss_dual: 1.071 loss_total: 10.029 batch_cost: 0.20106 sec reader_cost: 0.00029 sec ips: 39.78908 images/s eta: 0:22:51
    [05/29 18:26:27] ppgan.engine.trainer INFO: Iter: 13190/20000 lr: 9.996e-05 loss_promary: 9.151 loss_dual: 1.066 loss_total: 10.216 batch_cost: 0.19976 sec reader_cost: 0.00028 sec ips: 40.04885 images/s eta: 0:22:40
    [05/29 18:26:29] ppgan.engine.trainer INFO: Iter: 13200/20000 lr: 9.996e-05 loss_promary: 9.996 loss_dual: 1.128 loss_total: 11.124 batch_cost: 0.19811 sec reader_cost: 0.00028 sec ips: 40.38181 images/s eta: 0:22:27
    [05/29 18:26:31] ppgan.engine.trainer INFO: Iter: 13210/20000 lr: 9.996e-05 loss_promary: 9.234 loss_dual: 1.084 loss_total: 10.318 batch_cost: 0.20192 sec reader_cost: 0.00030 sec ips: 39.61950 images/s eta: 0:22:51
    [05/29 18:26:33] ppgan.engine.trainer INFO: Iter: 13220/20000 lr: 9.996e-05 loss_promary: 7.686 loss_dual: 0.965 loss_total: 8.651 batch_cost: 0.20269 sec reader_cost: 0.00030 sec ips: 39.46951 images/s eta: 0:22:54
    [05/29 18:26:35] ppgan.engine.trainer INFO: Iter: 13230/20000 lr: 9.996e-05 loss_promary: 9.290 loss_dual: 1.104 loss_total: 10.393 batch_cost: 0.20376 sec reader_cost: 0.00029 sec ips: 39.26184 images/s eta: 0:22:59
    [05/29 18:26:37] ppgan.engine.trainer INFO: Iter: 13240/20000 lr: 9.996e-05 loss_promary: 8.780 loss_dual: 1.118 loss_total: 9.898 batch_cost: 0.20370 sec reader_cost: 0.00029 sec ips: 39.27380 images/s eta: 0:22:56
    [05/29 18:26:39] ppgan.engine.trainer INFO: Iter: 13250/20000 lr: 9.996e-05 loss_promary: 12.926 loss_dual: 1.469 loss_total: 14.395 batch_cost: 0.20172 sec reader_cost: 0.00029 sec ips: 39.65977 images/s eta: 0:22:41
    [05/29 18:26:41] ppgan.engine.trainer INFO: Iter: 13260/20000 lr: 9.996e-05 loss_promary: 8.717 loss_dual: 1.101 loss_total: 9.818 batch_cost: 0.20217 sec reader_cost: 0.00029 sec ips: 39.57052 images/s eta: 0:22:42
    [05/29 18:26:43] ppgan.engine.trainer INFO: Iter: 13270/20000 lr: 9.996e-05 loss_promary: 9.768 loss_dual: 1.161 loss_total: 10.929 batch_cost: 0.20252 sec reader_cost: 0.00029 sec ips: 39.50201 images/s eta: 0:22:42
    [05/29 18:26:45] ppgan.engine.trainer INFO: Iter: 13280/20000 lr: 9.996e-05 loss_promary: 7.958 loss_dual: 1.005 loss_total: 8.963 batch_cost: 0.20274 sec reader_cost: 0.00028 sec ips: 39.45911 images/s eta: 0:22:42
    [05/29 18:26:47] ppgan.engine.trainer INFO: Iter: 13290/20000 lr: 9.996e-05 loss_promary: 8.377 loss_dual: 1.051 loss_total: 9.428 batch_cost: 0.20718 sec reader_cost: 0.00030 sec ips: 38.61365 images/s eta: 0:23:10
    [05/29 18:26:49] ppgan.engine.trainer INFO: Iter: 13300/20000 lr: 9.996e-05 loss_promary: 7.114 loss_dual: 0.842 loss_total: 7.956 batch_cost: 0.20045 sec reader_cost: 0.00028 sec ips: 39.91092 images/s eta: 0:22:22
    [05/29 18:26:51] ppgan.engine.trainer INFO: Iter: 13310/20000 lr: 9.996e-05 loss_promary: 9.479 loss_dual: 1.067 loss_total: 10.546 batch_cost: 0.22380 sec reader_cost: 0.00030 sec ips: 35.74647 images/s eta: 0:24:57
    [05/29 18:26:54] ppgan.engine.trainer INFO: Iter: 13320/20000 lr: 9.996e-05 loss_promary: 7.826 loss_dual: 0.953 loss_total: 8.779 batch_cost: 0.20307 sec reader_cost: 0.00030 sec ips: 39.39436 images/s eta: 0:22:36
    [05/29 18:26:56] ppgan.engine.trainer INFO: Iter: 13330/20000 lr: 9.996e-05 loss_promary: 6.486 loss_dual: 0.843 loss_total: 7.329 batch_cost: 0.20350 sec reader_cost: 0.00031 sec ips: 39.31203 images/s eta: 0:22:37
    [05/29 18:26:58] ppgan.engine.trainer INFO: Iter: 13340/20000 lr: 9.996e-05 loss_promary: 6.840 loss_dual: 0.893 loss_total: 7.733 batch_cost: 0.20175 sec reader_cost: 0.00030 sec ips: 39.65279 images/s eta: 0:22:23
    [05/29 18:27:00] ppgan.engine.trainer INFO: Iter: 13350/20000 lr: 9.996e-05 loss_promary: 8.526 loss_dual: 1.110 loss_total: 9.636 batch_cost: 0.20039 sec reader_cost: 0.00028 sec ips: 39.92216 images/s eta: 0:22:12
    [05/29 18:27:02] ppgan.engine.trainer INFO: Iter: 13360/20000 lr: 9.996e-05 loss_promary: 11.089 loss_dual: 1.210 loss_total: 12.299 batch_cost: 0.20248 sec reader_cost: 0.00029 sec ips: 39.51033 images/s eta: 0:22:24
    [05/29 18:27:04] ppgan.engine.trainer INFO: Iter: 13370/20000 lr: 9.996e-05 loss_promary: 8.228 loss_dual: 0.996 loss_total: 9.224 batch_cost: 0.20478 sec reader_cost: 0.00030 sec ips: 39.06600 images/s eta: 0:22:37
    [05/29 18:27:06] ppgan.engine.trainer INFO: Iter: 13380/20000 lr: 9.996e-05 loss_promary: 13.335 loss_dual: 1.455 loss_total: 14.790 batch_cost: 0.21494 sec reader_cost: 0.00031 sec ips: 37.21909 images/s eta: 0:23:42
    [05/29 18:27:08] ppgan.engine.trainer INFO: Iter: 13390/20000 lr: 9.996e-05 loss_promary: 9.435 loss_dual: 1.117 loss_total: 10.552 batch_cost: 0.20596 sec reader_cost: 0.00030 sec ips: 38.84204 images/s eta: 0:22:41
    [05/29 18:27:10] ppgan.engine.trainer INFO: Iter: 13400/20000 lr: 9.996e-05 loss_promary: 9.004 loss_dual: 1.076 loss_total: 10.080 batch_cost: 0.20937 sec reader_cost: 0.00030 sec ips: 38.21073 images/s eta: 0:23:01
    [05/29 18:27:12] ppgan.engine.trainer INFO: Iter: 13410/20000 lr: 9.996e-05 loss_promary: 7.771 loss_dual: 0.923 loss_total: 8.694 batch_cost: 0.21066 sec reader_cost: 0.00032 sec ips: 37.97649 images/s eta: 0:23:08
    [05/29 18:27:14] ppgan.engine.trainer INFO: Iter: 13420/20000 lr: 9.996e-05 loss_promary: 13.621 loss_dual: 1.461 loss_total: 15.082 batch_cost: 0.20568 sec reader_cost: 0.00029 sec ips: 38.89595 images/s eta: 0:22:33
    [05/29 18:27:16] ppgan.engine.trainer INFO: Iter: 13430/20000 lr: 9.996e-05 loss_promary: 10.626 loss_dual: 1.263 loss_total: 11.889 batch_cost: 0.21171 sec reader_cost: 0.00032 sec ips: 37.78746 images/s eta: 0:23:10
    [05/29 18:27:18] ppgan.engine.trainer INFO: Iter: 13440/20000 lr: 9.996e-05 loss_promary: 10.421 loss_dual: 1.182 loss_total: 11.603 batch_cost: 0.20445 sec reader_cost: 0.00041 sec ips: 39.12934 images/s eta: 0:22:21
    [05/29 18:27:21] ppgan.engine.trainer INFO: Iter: 13450/20000 lr: 9.996e-05 loss_promary: 10.531 loss_dual: 1.316 loss_total: 11.847 batch_cost: 0.26532 sec reader_cost: 0.03661 sec ips: 30.15270 images/s eta: 0:28:57
    [05/29 18:27:23] ppgan.engine.trainer INFO: Iter: 13460/20000 lr: 9.996e-05 loss_promary: 8.166 loss_dual: 1.046 loss_total: 9.212 batch_cost: 0.22524 sec reader_cost: 0.00031 sec ips: 35.51769 images/s eta: 0:24:33
    [05/29 18:27:25] ppgan.engine.trainer INFO: Iter: 13470/20000 lr: 9.996e-05 loss_promary: 9.353 loss_dual: 1.137 loss_total: 10.490 batch_cost: 0.21019 sec reader_cost: 0.00031 sec ips: 38.06067 images/s eta: 0:22:52
    [05/29 18:27:27] ppgan.engine.trainer INFO: Iter: 13480/20000 lr: 9.996e-05 loss_promary: 9.157 loss_dual: 1.090 loss_total: 10.247 batch_cost: 0.20709 sec reader_cost: 0.00031 sec ips: 38.63017 images/s eta: 0:22:30
    [05/29 18:27:29] ppgan.engine.trainer INFO: Iter: 13490/20000 lr: 9.996e-05 loss_promary: 11.711 loss_dual: 1.406 loss_total: 13.117 batch_cost: 0.20611 sec reader_cost: 0.00031 sec ips: 38.81388 images/s eta: 0:22:21
    [05/29 18:27:32] ppgan.engine.trainer INFO: Iter: 13500/20000 lr: 9.996e-05 loss_promary: 10.701 loss_dual: 1.233 loss_total: 11.934 batch_cost: 0.20874 sec reader_cost: 0.00030 sec ips: 38.32572 images/s eta: 0:22:36
    [05/29 18:27:34] ppgan.engine.trainer INFO: Iter: 13510/20000 lr: 9.996e-05 loss_promary: 8.681 loss_dual: 1.016 loss_total: 9.698 batch_cost: 0.20839 sec reader_cost: 0.00032 sec ips: 38.38898 images/s eta: 0:22:32
    [05/29 18:27:36] ppgan.engine.trainer INFO: Iter: 13520/20000 lr: 9.995e-05 loss_promary: 9.182 loss_dual: 1.082 loss_total: 10.264 batch_cost: 0.20747 sec reader_cost: 0.00030 sec ips: 38.55977 images/s eta: 0:22:24
    [05/29 18:27:38] ppgan.engine.trainer INFO: Iter: 13530/20000 lr: 9.995e-05 loss_promary: 10.831 loss_dual: 1.236 loss_total: 12.066 batch_cost: 0.21590 sec reader_cost: 0.00031 sec ips: 37.05399 images/s eta: 0:23:16
    [05/29 18:27:40] ppgan.engine.trainer INFO: Iter: 13540/20000 lr: 9.995e-05 loss_promary: 8.416 loss_dual: 1.012 loss_total: 9.428 batch_cost: 0.20908 sec reader_cost: 0.00032 sec ips: 38.26257 images/s eta: 0:22:30
    [05/29 18:27:42] ppgan.engine.trainer INFO: Iter: 13550/20000 lr: 9.995e-05 loss_promary: 11.829 loss_dual: 1.312 loss_total: 13.141 batch_cost: 0.21299 sec reader_cost: 0.00032 sec ips: 37.56046 images/s eta: 0:22:53
    [05/29 18:27:44] ppgan.engine.trainer INFO: Iter: 13560/20000 lr: 9.995e-05 loss_promary: 8.136 loss_dual: 1.135 loss_total: 9.272 batch_cost: 0.20810 sec reader_cost: 0.00031 sec ips: 38.44379 images/s eta: 0:22:20
    [05/29 18:27:46] ppgan.engine.trainer INFO: Iter: 13570/20000 lr: 9.995e-05 loss_promary: 7.574 loss_dual: 0.983 loss_total: 8.557 batch_cost: 0.21011 sec reader_cost: 0.00032 sec ips: 38.07456 images/s eta: 0:22:31
    [05/29 18:27:48] ppgan.engine.trainer INFO: Iter: 13580/20000 lr: 9.995e-05 loss_promary: 7.313 loss_dual: 0.951 loss_total: 8.264 batch_cost: 0.20649 sec reader_cost: 0.00030 sec ips: 38.74299 images/s eta: 0:22:05
    [05/29 18:27:50] ppgan.engine.trainer INFO: Iter: 13590/20000 lr: 9.995e-05 loss_promary: 8.691 loss_dual: 1.103 loss_total: 9.793 batch_cost: 0.20709 sec reader_cost: 0.00032 sec ips: 38.63110 images/s eta: 0:22:07
    [05/29 18:27:53] ppgan.engine.trainer INFO: Iter: 13600/20000 lr: 9.995e-05 loss_promary: 11.410 loss_dual: 1.323 loss_total: 12.732 batch_cost: 0.20717 sec reader_cost: 0.00031 sec ips: 38.61481 images/s eta: 0:22:05
    [05/29 18:27:55] ppgan.engine.trainer INFO: Iter: 13610/20000 lr: 9.995e-05 loss_promary: 8.096 loss_dual: 0.993 loss_total: 9.089 batch_cost: 0.24286 sec reader_cost: 0.00035 sec ips: 32.94138 images/s eta: 0:25:51
    [05/29 18:27:57] ppgan.engine.trainer INFO: Iter: 13620/20000 lr: 9.995e-05 loss_promary: 11.664 loss_dual: 1.264 loss_total: 12.928 batch_cost: 0.21826 sec reader_cost: 0.00032 sec ips: 36.65298 images/s eta: 0:23:12
    [05/29 18:27:59] ppgan.engine.trainer INFO: Iter: 13630/20000 lr: 9.995e-05 loss_promary: 10.959 loss_dual: 1.150 loss_total: 12.109 batch_cost: 0.21381 sec reader_cost: 0.00032 sec ips: 37.41655 images/s eta: 0:22:41
    [05/29 18:28:01] ppgan.engine.trainer INFO: Iter: 13640/20000 lr: 9.995e-05 loss_promary: 10.572 loss_dual: 1.295 loss_total: 11.867 batch_cost: 0.21675 sec reader_cost: 0.00033 sec ips: 36.90861 images/s eta: 0:22:58
    [05/29 18:28:04] ppgan.engine.trainer INFO: Iter: 13650/20000 lr: 9.995e-05 loss_promary: 7.363 loss_dual: 0.925 loss_total: 8.288 batch_cost: 0.20766 sec reader_cost: 0.00032 sec ips: 38.52531 images/s eta: 0:21:58
    [05/29 18:28:06] ppgan.engine.trainer INFO: Iter: 13660/20000 lr: 9.995e-05 loss_promary: 6.841 loss_dual: 0.976 loss_total: 7.817 batch_cost: 0.20999 sec reader_cost: 0.00033 sec ips: 38.09705 images/s eta: 0:22:11
    [05/29 18:28:08] ppgan.engine.trainer INFO: Iter: 13670/20000 lr: 9.995e-05 loss_promary: 8.813 loss_dual: 1.140 loss_total: 9.953 batch_cost: 0.20655 sec reader_cost: 0.00032 sec ips: 38.73218 images/s eta: 0:21:47
    [05/29 18:28:10] ppgan.engine.trainer INFO: Iter: 13680/20000 lr: 9.995e-05 loss_promary: 7.700 loss_dual: 1.022 loss_total: 8.722 batch_cost: 0.20673 sec reader_cost: 0.00030 sec ips: 38.69695 images/s eta: 0:21:46
    [05/29 18:28:12] ppgan.engine.trainer INFO: Iter: 13690/20000 lr: 9.995e-05 loss_promary: 7.564 loss_dual: 0.977 loss_total: 8.542 batch_cost: 0.20574 sec reader_cost: 0.00031 sec ips: 38.88379 images/s eta: 0:21:38
    [05/29 18:28:14] ppgan.engine.trainer INFO: Iter: 13700/20000 lr: 9.995e-05 loss_promary: 8.168 loss_dual: 0.965 loss_total: 9.133 batch_cost: 0.20478 sec reader_cost: 0.00030 sec ips: 39.06679 images/s eta: 0:21:30
    [05/29 18:28:16] ppgan.engine.trainer INFO: Iter: 13710/20000 lr: 9.995e-05 loss_promary: 9.061 loss_dual: 1.119 loss_total: 10.180 batch_cost: 0.21466 sec reader_cost: 0.00031 sec ips: 37.26893 images/s eta: 0:22:30
    [05/29 18:28:18] ppgan.engine.trainer INFO: Iter: 13720/20000 lr: 9.995e-05 loss_promary: 8.155 loss_dual: 1.001 loss_total: 9.157 batch_cost: 0.21070 sec reader_cost: 0.00030 sec ips: 37.96813 images/s eta: 0:22:03
    [05/29 18:28:20] ppgan.engine.trainer INFO: Iter: 13730/20000 lr: 9.995e-05 loss_promary: 10.906 loss_dual: 1.245 loss_total: 12.151 batch_cost: 0.20568 sec reader_cost: 0.00031 sec ips: 38.89616 images/s eta: 0:21:29
    [05/29 18:28:22] ppgan.engine.trainer INFO: Iter: 13740/20000 lr: 9.995e-05 loss_promary: 8.745 loss_dual: 1.028 loss_total: 9.773 batch_cost: 0.20831 sec reader_cost: 0.00030 sec ips: 38.40487 images/s eta: 0:21:44
    [05/29 18:28:24] ppgan.engine.trainer INFO: Iter: 13750/20000 lr: 9.995e-05 loss_promary: 10.325 loss_dual: 1.155 loss_total: 11.480 batch_cost: 0.20767 sec reader_cost: 0.00031 sec ips: 38.52268 images/s eta: 0:21:37
    [05/29 18:28:27] ppgan.engine.trainer INFO: Iter: 13760/20000 lr: 9.995e-05 loss_promary: 8.100 loss_dual: 0.979 loss_total: 9.078 batch_cost: 0.21436 sec reader_cost: 0.00031 sec ips: 37.32085 images/s eta: 0:22:17
    [05/29 18:28:29] ppgan.engine.trainer INFO: Iter: 13770/20000 lr: 9.995e-05 loss_promary: 9.072 loss_dual: 0.995 loss_total: 10.067 batch_cost: 0.21200 sec reader_cost: 0.00030 sec ips: 37.73625 images/s eta: 0:22:00
    [05/29 18:28:31] ppgan.engine.trainer INFO: Iter: 13780/20000 lr: 9.995e-05 loss_promary: 16.154 loss_dual: 1.721 loss_total: 17.875 batch_cost: 0.21072 sec reader_cost: 0.00032 sec ips: 37.96423 images/s eta: 0:21:50
    [05/29 18:28:33] ppgan.engine.trainer INFO: Iter: 13790/20000 lr: 9.995e-05 loss_promary: 12.826 loss_dual: 1.321 loss_total: 14.148 batch_cost: 0.20472 sec reader_cost: 0.00032 sec ips: 39.07693 images/s eta: 0:21:11
    [05/29 18:28:35] ppgan.engine.trainer INFO: Iter: 13800/20000 lr: 9.995e-05 loss_promary: 7.397 loss_dual: 0.861 loss_total: 8.257 batch_cost: 0.20613 sec reader_cost: 0.00030 sec ips: 38.81024 images/s eta: 0:21:18
    [05/29 18:28:37] ppgan.engine.trainer INFO: Iter: 13810/20000 lr: 9.995e-05 loss_promary: 11.764 loss_dual: 1.234 loss_total: 12.997 batch_cost: 0.20559 sec reader_cost: 0.00032 sec ips: 38.91277 images/s eta: 0:21:12
    [05/29 18:28:39] ppgan.engine.trainer INFO: Iter: 13820/20000 lr: 9.995e-05 loss_promary: 9.112 loss_dual: 0.982 loss_total: 10.095 batch_cost: 0.20895 sec reader_cost: 0.00032 sec ips: 38.28670 images/s eta: 0:21:31
    [05/29 18:28:41] ppgan.engine.trainer INFO: Iter: 13830/20000 lr: 9.995e-05 loss_promary: 10.599 loss_dual: 1.218 loss_total: 11.817 batch_cost: 0.20826 sec reader_cost: 0.00031 sec ips: 38.41319 images/s eta: 0:21:24
    [05/29 18:28:43] ppgan.engine.trainer INFO: Iter: 13840/20000 lr: 9.995e-05 loss_promary: 8.198 loss_dual: 1.054 loss_total: 9.252 batch_cost: 0.21309 sec reader_cost: 0.00041 sec ips: 37.54304 images/s eta: 0:21:52
    [05/29 18:28:45] ppgan.engine.trainer INFO: Iter: 13850/20000 lr: 9.995e-05 loss_promary: 12.623 loss_dual: 1.399 loss_total: 14.022 batch_cost: 0.21703 sec reader_cost: 0.00032 sec ips: 36.86104 images/s eta: 0:22:14
    [05/29 18:28:47] ppgan.engine.trainer INFO: Iter: 13860/20000 lr: 9.995e-05 loss_promary: 8.312 loss_dual: 1.033 loss_total: 9.345 batch_cost: 0.20320 sec reader_cost: 0.00031 sec ips: 39.36926 images/s eta: 0:20:47
    [05/29 18:28:50] ppgan.engine.trainer INFO: Iter: 13870/20000 lr: 9.995e-05 loss_promary: 7.620 loss_dual: 0.947 loss_total: 8.566 batch_cost: 0.20544 sec reader_cost: 0.00030 sec ips: 38.94060 images/s eta: 0:20:59
    [05/29 18:28:52] ppgan.engine.trainer INFO: Iter: 13880/20000 lr: 9.995e-05 loss_promary: 9.986 loss_dual: 1.303 loss_total: 11.289 batch_cost: 0.20658 sec reader_cost: 0.00031 sec ips: 38.72685 images/s eta: 0:21:04
    [05/29 18:28:54] ppgan.engine.trainer INFO: Iter: 13890/20000 lr: 9.995e-05 loss_promary: 11.980 loss_dual: 1.307 loss_total: 13.286 batch_cost: 0.20650 sec reader_cost: 0.00031 sec ips: 38.74033 images/s eta: 0:21:01
    [05/29 18:28:56] ppgan.engine.trainer INFO: Iter: 13900/20000 lr: 9.995e-05 loss_promary: 13.141 loss_dual: 1.456 loss_total: 14.597 batch_cost: 0.21294 sec reader_cost: 0.00056 sec ips: 37.56929 images/s eta: 0:21:38
    [05/29 18:28:58] ppgan.engine.trainer INFO: Iter: 13910/20000 lr: 9.995e-05 loss_promary: 10.052 loss_dual: 1.148 loss_total: 11.200 batch_cost: 0.21782 sec reader_cost: 0.00032 sec ips: 36.72768 images/s eta: 0:22:06
    [05/29 18:29:00] ppgan.engine.trainer INFO: Iter: 13920/20000 lr: 9.995e-05 loss_promary: 11.005 loss_dual: 1.246 loss_total: 12.251 batch_cost: 0.23435 sec reader_cost: 0.00032 sec ips: 34.13701 images/s eta: 0:23:44
    [05/29 18:29:02] ppgan.engine.trainer INFO: Iter: 13930/20000 lr: 9.995e-05 loss_promary: 8.656 loss_dual: 1.054 loss_total: 9.710 batch_cost: 0.20403 sec reader_cost: 0.00030 sec ips: 39.21020 images/s eta: 0:20:38
    [05/29 18:29:05] ppgan.engine.trainer INFO: Iter: 13940/20000 lr: 9.995e-05 loss_promary: 9.264 loss_dual: 1.051 loss_total: 10.315 batch_cost: 0.21598 sec reader_cost: 0.00033 sec ips: 37.04033 images/s eta: 0:21:48
    [05/29 18:29:07] ppgan.engine.trainer INFO: Iter: 13950/20000 lr: 9.995e-05 loss_promary: 6.152 loss_dual: 0.800 loss_total: 6.951 batch_cost: 0.21168 sec reader_cost: 0.00031 sec ips: 37.79305 images/s eta: 0:21:20
    [05/29 18:29:09] ppgan.engine.trainer INFO: Iter: 13960/20000 lr: 9.995e-05 loss_promary: 9.243 loss_dual: 1.011 loss_total: 10.254 batch_cost: 0.20746 sec reader_cost: 0.00031 sec ips: 38.56245 images/s eta: 0:20:53
    [05/29 18:29:11] ppgan.engine.trainer INFO: Iter: 13970/20000 lr: 9.995e-05 loss_promary: 8.862 loss_dual: 1.085 loss_total: 9.947 batch_cost: 0.20764 sec reader_cost: 0.00030 sec ips: 38.52776 images/s eta: 0:20:52
    [05/29 18:29:13] ppgan.engine.trainer INFO: Iter: 13980/20000 lr: 9.995e-05 loss_promary: 6.720 loss_dual: 0.861 loss_total: 7.581 batch_cost: 0.20673 sec reader_cost: 0.00031 sec ips: 38.69724 images/s eta: 0:20:44
    [05/29 18:29:15] ppgan.engine.trainer INFO: Iter: 13990/20000 lr: 9.995e-05 loss_promary: 9.563 loss_dual: 1.029 loss_total: 10.592 batch_cost: 0.21030 sec reader_cost: 0.00033 sec ips: 38.04026 images/s eta: 0:21:03
    [05/29 18:29:17] ppgan.engine.trainer INFO: Iter: 14000/20000 lr: 9.995e-05 loss_promary: 8.050 loss_dual: 0.937 loss_total: 8.987 batch_cost: 0.20326 sec reader_cost: 0.00030 sec ips: 39.35907 images/s eta: 0:20:19
    [05/29 18:29:17] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:29:19] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:29:21] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:29:22] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:29:24] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:29:26] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:29:27] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:29:29] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:29:31] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:29:33] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:29:34] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:29:36] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:29:38] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:29:39] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:29:41] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:29:43] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:29:44] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:29:46] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:29:48] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:29:49] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:29:51] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:29:53] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:29:55] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:29:56] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:29:58] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:30:00] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:30:01] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:30:03] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:30:05] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:30:07] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:30:08] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:30:10] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:30:12] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:30:13] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:30:15] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:30:17] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:30:18] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:30:20] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:30:22] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:30:24] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:30:25] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:30:27] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:30:29] ppgan.engine.trainer INFO: Metric psnr: 29.3986
    [05/29 18:30:29] ppgan.engine.trainer INFO: Metric ssim: 0.8001
    [05/29 18:30:31] ppgan.engine.trainer INFO: Iter: 14010/20000 lr: 9.995e-05 loss_promary: 10.112 loss_dual: 1.156 loss_total: 11.268 batch_cost: 0.20575 sec reader_cost: 0.00035 sec ips: 38.88300 images/s eta: 0:20:32
    [05/29 18:30:33] ppgan.engine.trainer INFO: Iter: 14020/20000 lr: 9.995e-05 loss_promary: 10.562 loss_dual: 1.167 loss_total: 11.729 batch_cost: 0.22208 sec reader_cost: 0.00030 sec ips: 36.02328 images/s eta: 0:22:08
    [05/29 18:30:35] ppgan.engine.trainer INFO: Iter: 14030/20000 lr: 9.995e-05 loss_promary: 8.603 loss_dual: 1.135 loss_total: 9.737 batch_cost: 0.21467 sec reader_cost: 0.00030 sec ips: 37.26697 images/s eta: 0:21:21
    [05/29 18:30:37] ppgan.engine.trainer INFO: Iter: 14040/20000 lr: 9.995e-05 loss_promary: 9.909 loss_dual: 1.199 loss_total: 11.108 batch_cost: 0.20783 sec reader_cost: 0.00029 sec ips: 38.49377 images/s eta: 0:20:38
    [05/29 18:30:39] ppgan.engine.trainer INFO: Iter: 14050/20000 lr: 9.995e-05 loss_promary: 10.997 loss_dual: 1.217 loss_total: 12.214 batch_cost: 0.20300 sec reader_cost: 0.00028 sec ips: 39.40793 images/s eta: 0:20:07
    [05/29 18:30:41] ppgan.engine.trainer INFO: Iter: 14060/20000 lr: 9.995e-05 loss_promary: 7.698 loss_dual: 0.980 loss_total: 8.678 batch_cost: 0.20098 sec reader_cost: 0.00029 sec ips: 39.80428 images/s eta: 0:19:53
    [05/29 18:30:43] ppgan.engine.trainer INFO: Iter: 14070/20000 lr: 9.995e-05 loss_promary: 10.729 loss_dual: 1.274 loss_total: 12.003 batch_cost: 0.20087 sec reader_cost: 0.00029 sec ips: 39.82625 images/s eta: 0:19:51
    [05/29 18:30:45] ppgan.engine.trainer INFO: Iter: 14080/20000 lr: 9.995e-05 loss_promary: 8.901 loss_dual: 1.176 loss_total: 10.077 batch_cost: 0.19969 sec reader_cost: 0.00029 sec ips: 40.06205 images/s eta: 0:19:42
    [05/29 18:30:47] ppgan.engine.trainer INFO: Iter: 14090/20000 lr: 9.995e-05 loss_promary: 9.272 loss_dual: 1.056 loss_total: 10.329 batch_cost: 0.19892 sec reader_cost: 0.00029 sec ips: 40.21754 images/s eta: 0:19:35
    [05/29 18:30:49] ppgan.engine.trainer INFO: Iter: 14100/20000 lr: 9.995e-05 loss_promary: 10.721 loss_dual: 1.186 loss_total: 11.907 batch_cost: 0.19963 sec reader_cost: 0.00029 sec ips: 40.07377 images/s eta: 0:19:37
    [05/29 18:30:51] ppgan.engine.trainer INFO: Iter: 14110/20000 lr: 9.995e-05 loss_promary: 10.173 loss_dual: 1.116 loss_total: 11.290 batch_cost: 0.20253 sec reader_cost: 0.00030 sec ips: 39.50104 images/s eta: 0:19:52
    [05/29 18:30:53] ppgan.engine.trainer INFO: Iter: 14120/20000 lr: 9.995e-05 loss_promary: 9.740 loss_dual: 1.099 loss_total: 10.839 batch_cost: 0.20200 sec reader_cost: 0.00029 sec ips: 39.60431 images/s eta: 0:19:47
    [05/29 18:30:56] ppgan.engine.trainer INFO: Iter: 14130/20000 lr: 9.995e-05 loss_promary: 6.798 loss_dual: 0.854 loss_total: 7.652 batch_cost: 0.20773 sec reader_cost: 0.00030 sec ips: 38.51219 images/s eta: 0:20:19
    [05/29 18:30:58] ppgan.engine.trainer INFO: Iter: 14140/20000 lr: 9.995e-05 loss_promary: 8.939 loss_dual: 1.010 loss_total: 9.949 batch_cost: 0.20927 sec reader_cost: 0.00030 sec ips: 38.22868 images/s eta: 0:20:26
    [05/29 18:31:00] ppgan.engine.trainer INFO: Iter: 14150/20000 lr: 9.995e-05 loss_promary: 12.361 loss_dual: 1.383 loss_total: 13.743 batch_cost: 0.20643 sec reader_cost: 0.00029 sec ips: 38.75355 images/s eta: 0:20:07
    [05/29 18:31:02] ppgan.engine.trainer INFO: Iter: 14160/20000 lr: 9.995e-05 loss_promary: 12.598 loss_dual: 1.497 loss_total: 14.096 batch_cost: 0.20125 sec reader_cost: 0.00030 sec ips: 39.75213 images/s eta: 0:19:35
    [05/29 18:31:04] ppgan.engine.trainer INFO: Iter: 14170/20000 lr: 9.995e-05 loss_promary: 10.670 loss_dual: 1.236 loss_total: 11.906 batch_cost: 0.20341 sec reader_cost: 0.00029 sec ips: 39.32923 images/s eta: 0:19:45
    [05/29 18:31:06] ppgan.engine.trainer INFO: Iter: 14180/20000 lr: 9.995e-05 loss_promary: 9.484 loss_dual: 1.031 loss_total: 10.515 batch_cost: 0.21538 sec reader_cost: 0.00028 sec ips: 37.14281 images/s eta: 0:20:53
    [05/29 18:31:08] ppgan.engine.trainer INFO: Iter: 14190/20000 lr: 9.995e-05 loss_promary: 8.015 loss_dual: 1.109 loss_total: 9.124 batch_cost: 0.19911 sec reader_cost: 0.00031 sec ips: 40.17804 images/s eta: 0:19:16
    [05/29 18:31:10] ppgan.engine.trainer INFO: Iter: 14200/20000 lr: 9.995e-05 loss_promary: 10.077 loss_dual: 1.188 loss_total: 11.265 batch_cost: 0.19923 sec reader_cost: 0.00030 sec ips: 40.15443 images/s eta: 0:19:15
    [05/29 18:31:12] ppgan.engine.trainer INFO: Iter: 14210/20000 lr: 9.995e-05 loss_promary: 10.249 loss_dual: 1.183 loss_total: 11.432 batch_cost: 0.20230 sec reader_cost: 0.00031 sec ips: 39.54554 images/s eta: 0:19:31
    [05/29 18:31:14] ppgan.engine.trainer INFO: Iter: 14220/20000 lr: 9.995e-05 loss_promary: 8.071 loss_dual: 0.987 loss_total: 9.058 batch_cost: 0.20112 sec reader_cost: 0.00030 sec ips: 39.77679 images/s eta: 0:19:22
    [05/29 18:31:16] ppgan.engine.trainer INFO: Iter: 14230/20000 lr: 9.995e-05 loss_promary: 8.376 loss_dual: 1.023 loss_total: 9.399 batch_cost: 0.20675 sec reader_cost: 0.00030 sec ips: 38.69445 images/s eta: 0:19:52
    [05/29 18:31:18] ppgan.engine.trainer INFO: Iter: 14240/20000 lr: 9.995e-05 loss_promary: 6.422 loss_dual: 0.825 loss_total: 7.247 batch_cost: 0.20238 sec reader_cost: 0.00028 sec ips: 39.52908 images/s eta: 0:19:25
    [05/29 18:31:20] ppgan.engine.trainer INFO: Iter: 14250/20000 lr: 9.995e-05 loss_promary: 8.361 loss_dual: 0.969 loss_total: 9.330 batch_cost: 0.21258 sec reader_cost: 0.00029 sec ips: 37.63330 images/s eta: 0:20:22
    [05/29 18:31:22] ppgan.engine.trainer INFO: Iter: 14260/20000 lr: 9.995e-05 loss_promary: 8.274 loss_dual: 1.032 loss_total: 9.306 batch_cost: 0.20469 sec reader_cost: 0.00030 sec ips: 39.08369 images/s eta: 0:19:34
    [05/29 18:31:24] ppgan.engine.trainer INFO: Iter: 14270/20000 lr: 9.995e-05 loss_promary: 11.031 loss_dual: 1.176 loss_total: 12.208 batch_cost: 0.19996 sec reader_cost: 0.00028 sec ips: 40.00776 images/s eta: 0:19:05
    [05/29 18:31:26] ppgan.engine.trainer INFO: Iter: 14280/20000 lr: 9.995e-05 loss_promary: 11.787 loss_dual: 1.265 loss_total: 13.052 batch_cost: 0.19551 sec reader_cost: 0.00033 sec ips: 40.91761 images/s eta: 0:18:38
    [05/29 18:31:29] ppgan.engine.trainer INFO: Iter: 14290/20000 lr: 9.995e-05 loss_promary: 8.528 loss_dual: 0.998 loss_total: 9.526 batch_cost: 0.25335 sec reader_cost: 0.04139 sec ips: 31.57708 images/s eta: 0:24:06
    [05/29 18:31:31] ppgan.engine.trainer INFO: Iter: 14300/20000 lr: 9.995e-05 loss_promary: 8.636 loss_dual: 1.002 loss_total: 9.639 batch_cost: 0.20016 sec reader_cost: 0.00028 sec ips: 39.96889 images/s eta: 0:19:00
    [05/29 18:31:33] ppgan.engine.trainer INFO: Iter: 14310/20000 lr: 9.995e-05 loss_promary: 10.645 loss_dual: 1.157 loss_total: 11.803 batch_cost: 0.19985 sec reader_cost: 0.00028 sec ips: 40.02981 images/s eta: 0:18:57
    [05/29 18:31:35] ppgan.engine.trainer INFO: Iter: 14320/20000 lr: 9.995e-05 loss_promary: 11.595 loss_dual: 1.234 loss_total: 12.829 batch_cost: 0.19919 sec reader_cost: 0.00029 sec ips: 40.16220 images/s eta: 0:18:51
    [05/29 18:31:37] ppgan.engine.trainer INFO: Iter: 14330/20000 lr: 9.995e-05 loss_promary: 9.189 loss_dual: 1.181 loss_total: 10.369 batch_cost: 0.20888 sec reader_cost: 0.00029 sec ips: 38.29899 images/s eta: 0:19:44
    [05/29 18:31:39] ppgan.engine.trainer INFO: Iter: 14340/20000 lr: 9.995e-05 loss_promary: 6.810 loss_dual: 0.932 loss_total: 7.742 batch_cost: 0.20487 sec reader_cost: 0.00028 sec ips: 39.04850 images/s eta: 0:19:19
    [05/29 18:31:41] ppgan.engine.trainer INFO: Iter: 14350/20000 lr: 9.995e-05 loss_promary: 10.124 loss_dual: 1.186 loss_total: 11.309 batch_cost: 0.20027 sec reader_cost: 0.00028 sec ips: 39.94526 images/s eta: 0:18:51
    [05/29 18:31:43] ppgan.engine.trainer INFO: Iter: 14360/20000 lr: 9.995e-05 loss_promary: 9.783 loss_dual: 1.361 loss_total: 11.144 batch_cost: 0.20405 sec reader_cost: 0.00028 sec ips: 39.20579 images/s eta: 0:19:10
    [05/29 18:31:45] ppgan.engine.trainer INFO: Iter: 14370/20000 lr: 9.995e-05 loss_promary: 9.238 loss_dual: 1.100 loss_total: 10.337 batch_cost: 0.20324 sec reader_cost: 0.00029 sec ips: 39.36219 images/s eta: 0:19:04
    [05/29 18:31:47] ppgan.engine.trainer INFO: Iter: 14380/20000 lr: 9.995e-05 loss_promary: 8.643 loss_dual: 1.149 loss_total: 9.792 batch_cost: 0.19976 sec reader_cost: 0.00028 sec ips: 40.04800 images/s eta: 0:18:42
    [05/29 18:31:49] ppgan.engine.trainer INFO: Iter: 14390/20000 lr: 9.995e-05 loss_promary: 10.373 loss_dual: 1.230 loss_total: 11.603 batch_cost: 0.19907 sec reader_cost: 0.00028 sec ips: 40.18778 images/s eta: 0:18:36
    [05/29 18:31:51] ppgan.engine.trainer INFO: Iter: 14400/20000 lr: 9.995e-05 loss_promary: 8.257 loss_dual: 1.037 loss_total: 9.294 batch_cost: 0.20072 sec reader_cost: 0.00029 sec ips: 39.85737 images/s eta: 0:18:44
    [05/29 18:31:53] ppgan.engine.trainer INFO: Iter: 14410/20000 lr: 9.995e-05 loss_promary: 9.245 loss_dual: 1.147 loss_total: 10.392 batch_cost: 0.20300 sec reader_cost: 0.00029 sec ips: 39.40844 images/s eta: 0:18:54
    [05/29 18:31:55] ppgan.engine.trainer INFO: Iter: 14420/20000 lr: 9.995e-05 loss_promary: 7.452 loss_dual: 0.920 loss_total: 8.373 batch_cost: 0.20183 sec reader_cost: 0.00030 sec ips: 39.63735 images/s eta: 0:18:46
    [05/29 18:31:57] ppgan.engine.trainer INFO: Iter: 14430/20000 lr: 9.995e-05 loss_promary: 12.299 loss_dual: 1.327 loss_total: 13.626 batch_cost: 0.20060 sec reader_cost: 0.00029 sec ips: 39.87980 images/s eta: 0:18:37
    [05/29 18:31:59] ppgan.engine.trainer INFO: Iter: 14440/20000 lr: 9.995e-05 loss_promary: 8.736 loss_dual: 1.016 loss_total: 9.752 batch_cost: 0.20003 sec reader_cost: 0.00029 sec ips: 39.99411 images/s eta: 0:18:32
    [05/29 18:32:01] ppgan.engine.trainer INFO: Iter: 14450/20000 lr: 9.995e-05 loss_promary: 7.941 loss_dual: 0.943 loss_total: 8.883 batch_cost: 0.20272 sec reader_cost: 0.00029 sec ips: 39.46401 images/s eta: 0:18:45
    [05/29 18:32:03] ppgan.engine.trainer INFO: Iter: 14460/20000 lr: 9.995e-05 loss_promary: 10.497 loss_dual: 1.180 loss_total: 11.676 batch_cost: 0.20078 sec reader_cost: 0.00029 sec ips: 39.84366 images/s eta: 0:18:32
    [05/29 18:32:05] ppgan.engine.trainer INFO: Iter: 14470/20000 lr: 9.995e-05 loss_promary: 10.983 loss_dual: 1.164 loss_total: 12.147 batch_cost: 0.19853 sec reader_cost: 0.00028 sec ips: 40.29622 images/s eta: 0:18:17
    [05/29 18:32:07] ppgan.engine.trainer INFO: Iter: 14480/20000 lr: 9.995e-05 loss_promary: 9.368 loss_dual: 1.091 loss_total: 10.459 batch_cost: 0.20001 sec reader_cost: 0.00029 sec ips: 39.99823 images/s eta: 0:18:24
    [05/29 18:32:09] ppgan.engine.trainer INFO: Iter: 14490/20000 lr: 9.995e-05 loss_promary: 9.866 loss_dual: 1.154 loss_total: 11.020 batch_cost: 0.20937 sec reader_cost: 0.00029 sec ips: 38.21030 images/s eta: 0:19:13
    [05/29 18:32:11] ppgan.engine.trainer INFO: Iter: 14500/20000 lr: 9.995e-05 loss_promary: 7.038 loss_dual: 0.841 loss_total: 7.879 batch_cost: 0.20565 sec reader_cost: 0.00031 sec ips: 38.90059 images/s eta: 0:18:51
    [05/29 18:32:13] ppgan.engine.trainer INFO: Iter: 14510/20000 lr: 9.995e-05 loss_promary: 6.235 loss_dual: 0.791 loss_total: 7.027 batch_cost: 0.20152 sec reader_cost: 0.00031 sec ips: 39.69757 images/s eta: 0:18:26
    [05/29 18:32:15] ppgan.engine.trainer INFO: Iter: 14520/20000 lr: 9.995e-05 loss_promary: 10.434 loss_dual: 1.114 loss_total: 11.548 batch_cost: 0.20119 sec reader_cost: 0.00028 sec ips: 39.76373 images/s eta: 0:18:22
    [05/29 18:32:17] ppgan.engine.trainer INFO: Iter: 14530/20000 lr: 9.995e-05 loss_promary: 10.619 loss_dual: 1.158 loss_total: 11.777 batch_cost: 0.19892 sec reader_cost: 0.00029 sec ips: 40.21690 images/s eta: 0:18:08
    [05/29 18:32:19] ppgan.engine.trainer INFO: Iter: 14540/20000 lr: 9.995e-05 loss_promary: 10.877 loss_dual: 1.173 loss_total: 12.049 batch_cost: 0.19678 sec reader_cost: 0.00027 sec ips: 40.65466 images/s eta: 0:17:54
    [05/29 18:32:21] ppgan.engine.trainer INFO: Iter: 14550/20000 lr: 9.995e-05 loss_promary: 8.888 loss_dual: 1.129 loss_total: 10.017 batch_cost: 0.20395 sec reader_cost: 0.00028 sec ips: 39.22608 images/s eta: 0:18:31
    [05/29 18:32:23] ppgan.engine.trainer INFO: Iter: 14560/20000 lr: 9.995e-05 loss_promary: 9.001 loss_dual: 1.020 loss_total: 10.022 batch_cost: 0.20273 sec reader_cost: 0.00029 sec ips: 39.46211 images/s eta: 0:18:22
    [05/29 18:32:25] ppgan.engine.trainer INFO: Iter: 14570/20000 lr: 9.995e-05 loss_promary: 7.810 loss_dual: 0.947 loss_total: 8.758 batch_cost: 0.20808 sec reader_cost: 0.00031 sec ips: 38.44647 images/s eta: 0:18:49
    [05/29 18:32:27] ppgan.engine.trainer INFO: Iter: 14580/20000 lr: 9.995e-05 loss_promary: 13.251 loss_dual: 1.400 loss_total: 14.651 batch_cost: 0.19742 sec reader_cost: 0.00028 sec ips: 40.52327 images/s eta: 0:17:50
    [05/29 18:32:29] ppgan.engine.trainer INFO: Iter: 14590/20000 lr: 9.995e-05 loss_promary: 7.140 loss_dual: 0.972 loss_total: 8.112 batch_cost: 0.19672 sec reader_cost: 0.00028 sec ips: 40.66701 images/s eta: 0:17:44
    [05/29 18:32:31] ppgan.engine.trainer INFO: Iter: 14600/20000 lr: 9.995e-05 loss_promary: 10.515 loss_dual: 1.206 loss_total: 11.722 batch_cost: 0.20126 sec reader_cost: 0.00029 sec ips: 39.74893 images/s eta: 0:18:06
    [05/29 18:32:33] ppgan.engine.trainer INFO: Iter: 14610/20000 lr: 9.995e-05 loss_promary: 8.256 loss_dual: 1.036 loss_total: 9.292 batch_cost: 0.20088 sec reader_cost: 0.00029 sec ips: 39.82405 images/s eta: 0:18:02
    [05/29 18:32:35] ppgan.engine.trainer INFO: Iter: 14620/20000 lr: 9.995e-05 loss_promary: 9.778 loss_dual: 1.108 loss_total: 10.886 batch_cost: 0.19950 sec reader_cost: 0.00028 sec ips: 40.09953 images/s eta: 0:17:53
    [05/29 18:32:37] ppgan.engine.trainer INFO: Iter: 14630/20000 lr: 9.995e-05 loss_promary: 6.284 loss_dual: 0.816 loss_total: 7.100 batch_cost: 0.20246 sec reader_cost: 0.00028 sec ips: 39.51398 images/s eta: 0:18:07
    [05/29 18:32:39] ppgan.engine.trainer INFO: Iter: 14640/20000 lr: 9.995e-05 loss_promary: 9.910 loss_dual: 1.137 loss_total: 11.047 batch_cost: 0.20249 sec reader_cost: 0.00029 sec ips: 39.50741 images/s eta: 0:18:05
    [05/29 18:32:42] ppgan.engine.trainer INFO: Iter: 14650/20000 lr: 9.995e-05 loss_promary: 10.016 loss_dual: 1.089 loss_total: 11.105 batch_cost: 0.24251 sec reader_cost: 0.00031 sec ips: 32.98877 images/s eta: 0:21:37
    [05/29 18:32:44] ppgan.engine.trainer INFO: Iter: 14660/20000 lr: 9.995e-05 loss_promary: 10.067 loss_dual: 1.256 loss_total: 11.323 batch_cost: 0.20307 sec reader_cost: 0.00030 sec ips: 39.39550 images/s eta: 0:18:04
    [05/29 18:32:46] ppgan.engine.trainer INFO: Iter: 14670/20000 lr: 9.995e-05 loss_promary: 13.600 loss_dual: 1.533 loss_total: 15.133 batch_cost: 0.20310 sec reader_cost: 0.00030 sec ips: 39.38905 images/s eta: 0:18:02
    [05/29 18:32:48] ppgan.engine.trainer INFO: Iter: 14680/20000 lr: 9.995e-05 loss_promary: 8.491 loss_dual: 1.191 loss_total: 9.683 batch_cost: 0.20376 sec reader_cost: 0.00031 sec ips: 39.26138 images/s eta: 0:18:04
    [05/29 18:32:50] ppgan.engine.trainer INFO: Iter: 14690/20000 lr: 9.995e-05 loss_promary: 9.058 loss_dual: 1.326 loss_total: 10.384 batch_cost: 0.20275 sec reader_cost: 0.00030 sec ips: 39.45830 images/s eta: 0:17:56
    [05/29 18:32:52] ppgan.engine.trainer INFO: Iter: 14700/20000 lr: 9.995e-05 loss_promary: 7.975 loss_dual: 1.013 loss_total: 8.988 batch_cost: 0.20424 sec reader_cost: 0.00029 sec ips: 39.17005 images/s eta: 0:18:02
    [05/29 18:32:54] ppgan.engine.trainer INFO: Iter: 14710/20000 lr: 9.995e-05 loss_promary: 7.992 loss_dual: 0.979 loss_total: 8.971 batch_cost: 0.20214 sec reader_cost: 0.00029 sec ips: 39.57596 images/s eta: 0:17:49
    [05/29 18:32:56] ppgan.engine.trainer INFO: Iter: 14720/20000 lr: 9.995e-05 loss_promary: 7.922 loss_dual: 0.993 loss_total: 8.915 batch_cost: 0.20495 sec reader_cost: 0.00030 sec ips: 39.03387 images/s eta: 0:18:02
    [05/29 18:32:58] ppgan.engine.trainer INFO: Iter: 14730/20000 lr: 9.995e-05 loss_promary: 11.638 loss_dual: 1.306 loss_total: 12.944 batch_cost: 0.20215 sec reader_cost: 0.00029 sec ips: 39.57456 images/s eta: 0:17:45
    [05/29 18:33:00] ppgan.engine.trainer INFO: Iter: 14740/20000 lr: 9.995e-05 loss_promary: 7.354 loss_dual: 0.944 loss_total: 8.299 batch_cost: 0.20548 sec reader_cost: 0.00030 sec ips: 38.93244 images/s eta: 0:18:00
    [05/29 18:33:02] ppgan.engine.trainer INFO: Iter: 14750/20000 lr: 9.995e-05 loss_promary: 6.999 loss_dual: 1.001 loss_total: 8.001 batch_cost: 0.20235 sec reader_cost: 0.00030 sec ips: 39.53566 images/s eta: 0:17:42
    [05/29 18:33:04] ppgan.engine.trainer INFO: Iter: 14760/20000 lr: 9.995e-05 loss_promary: 10.849 loss_dual: 1.275 loss_total: 12.124 batch_cost: 0.20300 sec reader_cost: 0.00030 sec ips: 39.40943 images/s eta: 0:17:43
    [05/29 18:33:06] ppgan.engine.trainer INFO: Iter: 14770/20000 lr: 9.995e-05 loss_promary: 8.971 loss_dual: 1.065 loss_total: 10.036 batch_cost: 0.20164 sec reader_cost: 0.00030 sec ips: 39.67544 images/s eta: 0:17:34
    [05/29 18:33:08] ppgan.engine.trainer INFO: Iter: 14780/20000 lr: 9.995e-05 loss_promary: 11.866 loss_dual: 1.420 loss_total: 13.286 batch_cost: 0.20026 sec reader_cost: 0.00029 sec ips: 39.94828 images/s eta: 0:17:25
    [05/29 18:33:10] ppgan.engine.trainer INFO: Iter: 14790/20000 lr: 9.995e-05 loss_promary: 9.012 loss_dual: 1.151 loss_total: 10.164 batch_cost: 0.19973 sec reader_cost: 0.00029 sec ips: 40.05492 images/s eta: 0:17:20
    [05/29 18:33:12] ppgan.engine.trainer INFO: Iter: 14800/20000 lr: 9.995e-05 loss_promary: 11.921 loss_dual: 1.387 loss_total: 13.308 batch_cost: 0.20845 sec reader_cost: 0.00030 sec ips: 38.37924 images/s eta: 0:18:03
    [05/29 18:33:14] ppgan.engine.trainer INFO: Iter: 14810/20000 lr: 9.995e-05 loss_promary: 13.424 loss_dual: 1.481 loss_total: 14.905 batch_cost: 0.20794 sec reader_cost: 0.00031 sec ips: 38.47269 images/s eta: 0:17:59
    [05/29 18:33:16] ppgan.engine.trainer INFO: Iter: 14820/20000 lr: 9.995e-05 loss_promary: 8.935 loss_dual: 1.098 loss_total: 10.033 batch_cost: 0.19820 sec reader_cost: 0.00028 sec ips: 40.36246 images/s eta: 0:17:06
    [05/29 18:33:18] ppgan.engine.trainer INFO: Iter: 14830/20000 lr: 9.995e-05 loss_promary: 8.043 loss_dual: 0.912 loss_total: 8.956 batch_cost: 0.20164 sec reader_cost: 0.00030 sec ips: 39.67506 images/s eta: 0:17:22
    [05/29 18:33:20] ppgan.engine.trainer INFO: Iter: 14840/20000 lr: 9.995e-05 loss_promary: 8.854 loss_dual: 1.040 loss_total: 9.894 batch_cost: 0.19939 sec reader_cost: 0.00029 sec ips: 40.12269 images/s eta: 0:17:08
    [05/29 18:33:22] ppgan.engine.trainer INFO: Iter: 14850/20000 lr: 9.995e-05 loss_promary: 9.458 loss_dual: 1.119 loss_total: 10.576 batch_cost: 0.20213 sec reader_cost: 0.00029 sec ips: 39.57760 images/s eta: 0:17:20
    [05/29 18:33:24] ppgan.engine.trainer INFO: Iter: 14860/20000 lr: 9.995e-05 loss_promary: 10.007 loss_dual: 1.179 loss_total: 11.187 batch_cost: 0.19791 sec reader_cost: 0.00031 sec ips: 40.42277 images/s eta: 0:16:57
    [05/29 18:33:26] ppgan.engine.trainer INFO: Iter: 14870/20000 lr: 9.995e-05 loss_promary: 9.437 loss_dual: 1.052 loss_total: 10.490 batch_cost: 0.19666 sec reader_cost: 0.00029 sec ips: 40.67867 images/s eta: 0:16:48
    [05/29 18:33:28] ppgan.engine.trainer INFO: Iter: 14880/20000 lr: 9.995e-05 loss_promary: 7.477 loss_dual: 0.861 loss_total: 8.338 batch_cost: 0.20003 sec reader_cost: 0.00030 sec ips: 39.99397 images/s eta: 0:17:04
    [05/29 18:33:30] ppgan.engine.trainer INFO: Iter: 14890/20000 lr: 9.995e-05 loss_promary: 10.703 loss_dual: 1.183 loss_total: 11.886 batch_cost: 0.20695 sec reader_cost: 0.00033 sec ips: 38.65652 images/s eta: 0:17:37
    [05/29 18:33:33] ppgan.engine.trainer INFO: Iter: 14900/20000 lr: 9.995e-05 loss_promary: 8.871 loss_dual: 1.025 loss_total: 9.896 batch_cost: 0.21094 sec reader_cost: 0.00031 sec ips: 37.92497 images/s eta: 0:17:55
    [05/29 18:33:35] ppgan.engine.trainer INFO: Iter: 14910/20000 lr: 9.995e-05 loss_promary: 9.941 loss_dual: 1.086 loss_total: 11.027 batch_cost: 0.20317 sec reader_cost: 0.00030 sec ips: 39.37616 images/s eta: 0:17:14
    [05/29 18:33:37] ppgan.engine.trainer INFO: Iter: 14920/20000 lr: 9.995e-05 loss_promary: 8.276 loss_dual: 0.960 loss_total: 9.236 batch_cost: 0.20126 sec reader_cost: 0.00029 sec ips: 39.74875 images/s eta: 0:17:02
    [05/29 18:33:39] ppgan.engine.trainer INFO: Iter: 14930/20000 lr: 9.995e-05 loss_promary: 12.820 loss_dual: 1.344 loss_total: 14.164 batch_cost: 0.19993 sec reader_cost: 0.00029 sec ips: 40.01367 images/s eta: 0:16:53
    [05/29 18:33:41] ppgan.engine.trainer INFO: Iter: 14940/20000 lr: 9.994e-05 loss_promary: 9.503 loss_dual: 1.056 loss_total: 10.559 batch_cost: 0.20088 sec reader_cost: 0.00029 sec ips: 39.82521 images/s eta: 0:16:56
    [05/29 18:33:43] ppgan.engine.trainer INFO: Iter: 14950/20000 lr: 9.994e-05 loss_promary: 8.747 loss_dual: 1.047 loss_total: 9.794 batch_cost: 0.20319 sec reader_cost: 0.00030 sec ips: 39.37143 images/s eta: 0:17:06
    [05/29 18:33:45] ppgan.engine.trainer INFO: Iter: 14960/20000 lr: 9.994e-05 loss_promary: 8.836 loss_dual: 1.131 loss_total: 9.967 batch_cost: 0.22361 sec reader_cost: 0.00030 sec ips: 35.77664 images/s eta: 0:18:46
    [05/29 18:33:47] ppgan.engine.trainer INFO: Iter: 14970/20000 lr: 9.994e-05 loss_promary: 7.329 loss_dual: 0.950 loss_total: 8.280 batch_cost: 0.21361 sec reader_cost: 0.00031 sec ips: 37.45095 images/s eta: 0:17:54
    [05/29 18:33:49] ppgan.engine.trainer INFO: Iter: 14980/20000 lr: 9.994e-05 loss_promary: 8.708 loss_dual: 1.031 loss_total: 9.739 batch_cost: 0.23477 sec reader_cost: 0.00034 sec ips: 34.07602 images/s eta: 0:19:38
    [05/29 18:33:52] ppgan.engine.trainer INFO: Iter: 14990/20000 lr: 9.994e-05 loss_promary: 8.071 loss_dual: 1.008 loss_total: 9.079 batch_cost: 0.23168 sec reader_cost: 0.00032 sec ips: 34.53012 images/s eta: 0:19:20
    [05/29 18:33:54] ppgan.engine.trainer INFO: Iter: 15000/20000 lr: 9.994e-05 loss_promary: 9.766 loss_dual: 1.084 loss_total: 10.850 batch_cost: 0.23203 sec reader_cost: 0.00034 sec ips: 34.47803 images/s eta: 0:19:20
    [05/29 18:33:54] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:33:56] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:33:58] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:34:00] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:34:01] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:34:03] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:34:05] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:34:07] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:34:08] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:34:10] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:34:12] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:34:13] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:34:15] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:34:17] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:34:19] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:34:20] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:34:22] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:34:24] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:34:25] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:34:27] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:34:29] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:34:31] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:34:32] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:34:34] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:34:36] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:34:37] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:34:39] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:34:41] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:34:42] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:34:44] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:34:46] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:34:47] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:34:49] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:34:51] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:34:53] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:34:54] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:34:56] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:34:58] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:34:59] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:35:01] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:35:03] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:35:05] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:35:06] ppgan.engine.trainer INFO: Metric psnr: 29.3807
    [05/29 18:35:06] ppgan.engine.trainer INFO: Metric ssim: 0.7985
    [05/29 18:35:09] ppgan.engine.trainer INFO: Iter: 15010/20000 lr: 9.994e-05 loss_promary: 7.440 loss_dual: 0.993 loss_total: 8.434 batch_cost: 0.21269 sec reader_cost: 0.00035 sec ips: 37.61300 images/s eta: 0:17:41
    [05/29 18:35:11] ppgan.engine.trainer INFO: Iter: 15020/20000 lr: 9.994e-05 loss_promary: 8.244 loss_dual: 1.344 loss_total: 9.588 batch_cost: 0.22772 sec reader_cost: 0.00030 sec ips: 35.13117 images/s eta: 0:18:54
    [05/29 18:35:13] ppgan.engine.trainer INFO: Iter: 15030/20000 lr: 9.994e-05 loss_promary: 10.118 loss_dual: 1.499 loss_total: 11.617 batch_cost: 0.22857 sec reader_cost: 0.00031 sec ips: 35.00008 images/s eta: 0:18:55
    [05/29 18:35:15] ppgan.engine.trainer INFO: Iter: 15040/20000 lr: 9.994e-05 loss_promary: 11.559 loss_dual: 1.367 loss_total: 12.926 batch_cost: 0.22623 sec reader_cost: 0.00031 sec ips: 35.36238 images/s eta: 0:18:42
    [05/29 18:35:18] ppgan.engine.trainer INFO: Iter: 15050/20000 lr: 9.994e-05 loss_promary: 10.487 loss_dual: 1.217 loss_total: 11.704 batch_cost: 0.22398 sec reader_cost: 0.00031 sec ips: 35.71798 images/s eta: 0:18:28
    [05/29 18:35:20] ppgan.engine.trainer INFO: Iter: 15060/20000 lr: 9.994e-05 loss_promary: 8.998 loss_dual: 1.140 loss_total: 10.138 batch_cost: 0.23517 sec reader_cost: 0.00032 sec ips: 34.01729 images/s eta: 0:19:21
    [05/29 18:35:22] ppgan.engine.trainer INFO: Iter: 15070/20000 lr: 9.994e-05 loss_promary: 10.374 loss_dual: 1.255 loss_total: 11.629 batch_cost: 0.20377 sec reader_cost: 0.00030 sec ips: 39.25948 images/s eta: 0:16:44
    [05/29 18:35:24] ppgan.engine.trainer INFO: Iter: 15080/20000 lr: 9.994e-05 loss_promary: 10.711 loss_dual: 1.196 loss_total: 11.907 batch_cost: 0.20447 sec reader_cost: 0.00029 sec ips: 39.12496 images/s eta: 0:16:46
    [05/29 18:35:26] ppgan.engine.trainer INFO: Iter: 15090/20000 lr: 9.994e-05 loss_promary: 8.129 loss_dual: 0.967 loss_total: 9.096 batch_cost: 0.19740 sec reader_cost: 0.00028 sec ips: 40.52650 images/s eta: 0:16:09
    [05/29 18:35:28] ppgan.engine.trainer INFO: Iter: 15100/20000 lr: 9.994e-05 loss_promary: 9.412 loss_dual: 1.091 loss_total: 10.503 batch_cost: 0.19834 sec reader_cost: 0.00027 sec ips: 40.33514 images/s eta: 0:16:11
    [05/29 18:35:30] ppgan.engine.trainer INFO: Iter: 15110/20000 lr: 9.994e-05 loss_promary: 6.952 loss_dual: 0.854 loss_total: 7.807 batch_cost: 0.19949 sec reader_cost: 0.00028 sec ips: 40.10134 images/s eta: 0:16:15
    [05/29 18:35:32] ppgan.engine.trainer INFO: Iter: 15120/20000 lr: 9.994e-05 loss_promary: 8.225 loss_dual: 1.040 loss_total: 9.264 batch_cost: 0.19803 sec reader_cost: 0.00036 sec ips: 40.39711 images/s eta: 0:16:06
    [05/29 18:35:35] ppgan.engine.trainer INFO: Iter: 15130/20000 lr: 9.994e-05 loss_promary: 8.237 loss_dual: 1.028 loss_total: 9.265 batch_cost: 0.25965 sec reader_cost: 0.04032 sec ips: 30.81056 images/s eta: 0:21:04
    [05/29 18:35:37] ppgan.engine.trainer INFO: Iter: 15140/20000 lr: 9.994e-05 loss_promary: 6.395 loss_dual: 0.780 loss_total: 7.175 batch_cost: 0.20277 sec reader_cost: 0.00030 sec ips: 39.45275 images/s eta: 0:16:25
    [05/29 18:35:39] ppgan.engine.trainer INFO: Iter: 15150/20000 lr: 9.994e-05 loss_promary: 7.573 loss_dual: 0.881 loss_total: 8.454 batch_cost: 0.20570 sec reader_cost: 0.00030 sec ips: 38.89133 images/s eta: 0:16:37
    [05/29 18:35:41] ppgan.engine.trainer INFO: Iter: 15160/20000 lr: 9.994e-05 loss_promary: 7.196 loss_dual: 0.881 loss_total: 8.077 batch_cost: 0.20326 sec reader_cost: 0.00030 sec ips: 39.35790 images/s eta: 0:16:23
    [05/29 18:35:43] ppgan.engine.trainer INFO: Iter: 15170/20000 lr: 9.994e-05 loss_promary: 12.359 loss_dual: 1.274 loss_total: 13.633 batch_cost: 0.20245 sec reader_cost: 0.00031 sec ips: 39.51525 images/s eta: 0:16:17
    [05/29 18:35:45] ppgan.engine.trainer INFO: Iter: 15180/20000 lr: 9.994e-05 loss_promary: 8.881 loss_dual: 0.996 loss_total: 9.877 batch_cost: 0.21162 sec reader_cost: 0.00031 sec ips: 37.80417 images/s eta: 0:16:59
    [05/29 18:35:47] ppgan.engine.trainer INFO: Iter: 15190/20000 lr: 9.994e-05 loss_promary: 9.280 loss_dual: 1.022 loss_total: 10.302 batch_cost: 0.21522 sec reader_cost: 0.00032 sec ips: 37.17209 images/s eta: 0:17:15
    [05/29 18:35:49] ppgan.engine.trainer INFO: Iter: 15200/20000 lr: 9.994e-05 loss_promary: 10.015 loss_dual: 1.103 loss_total: 11.119 batch_cost: 0.20632 sec reader_cost: 0.00031 sec ips: 38.77394 images/s eta: 0:16:30
    [05/29 18:35:51] ppgan.engine.trainer INFO: Iter: 15210/20000 lr: 9.994e-05 loss_promary: 7.438 loss_dual: 0.902 loss_total: 8.340 batch_cost: 0.21486 sec reader_cost: 0.00031 sec ips: 37.23313 images/s eta: 0:17:09
    [05/29 18:35:53] ppgan.engine.trainer INFO: Iter: 15220/20000 lr: 9.994e-05 loss_promary: 10.627 loss_dual: 1.243 loss_total: 11.870 batch_cost: 0.20800 sec reader_cost: 0.00030 sec ips: 38.46230 images/s eta: 0:16:34
    [05/29 18:35:55] ppgan.engine.trainer INFO: Iter: 15230/20000 lr: 9.994e-05 loss_promary: 10.482 loss_dual: 1.189 loss_total: 11.672 batch_cost: 0.20334 sec reader_cost: 0.00030 sec ips: 39.34322 images/s eta: 0:16:09
    [05/29 18:35:57] ppgan.engine.trainer INFO: Iter: 15240/20000 lr: 9.994e-05 loss_promary: 7.513 loss_dual: 0.924 loss_total: 8.437 batch_cost: 0.20138 sec reader_cost: 0.00030 sec ips: 39.72504 images/s eta: 0:15:58
    [05/29 18:35:59] ppgan.engine.trainer INFO: Iter: 15250/20000 lr: 9.994e-05 loss_promary: 11.687 loss_dual: 1.305 loss_total: 12.992 batch_cost: 0.19917 sec reader_cost: 0.00029 sec ips: 40.16578 images/s eta: 0:15:46
    [05/29 18:36:01] ppgan.engine.trainer INFO: Iter: 15260/20000 lr: 9.994e-05 loss_promary: 6.432 loss_dual: 0.817 loss_total: 7.249 batch_cost: 0.20282 sec reader_cost: 0.00029 sec ips: 39.44396 images/s eta: 0:16:01
    [05/29 18:36:03] ppgan.engine.trainer INFO: Iter: 15270/20000 lr: 9.994e-05 loss_promary: 10.378 loss_dual: 1.192 loss_total: 11.570 batch_cost: 0.19820 sec reader_cost: 0.00029 sec ips: 40.36395 images/s eta: 0:15:37
    [05/29 18:36:05] ppgan.engine.trainer INFO: Iter: 15280/20000 lr: 9.994e-05 loss_promary: 8.917 loss_dual: 1.142 loss_total: 10.059 batch_cost: 0.19882 sec reader_cost: 0.00029 sec ips: 40.23829 images/s eta: 0:15:38
    [05/29 18:36:07] ppgan.engine.trainer INFO: Iter: 15290/20000 lr: 9.994e-05 loss_promary: 9.845 loss_dual: 1.203 loss_total: 11.048 batch_cost: 0.19696 sec reader_cost: 0.00028 sec ips: 40.61754 images/s eta: 0:15:27
    [05/29 18:36:09] ppgan.engine.trainer INFO: Iter: 15300/20000 lr: 9.994e-05 loss_promary: 9.796 loss_dual: 1.261 loss_total: 11.057 batch_cost: 0.19843 sec reader_cost: 0.00030 sec ips: 40.31703 images/s eta: 0:15:32
    [05/29 18:36:11] ppgan.engine.trainer INFO: Iter: 15310/20000 lr: 9.994e-05 loss_promary: 7.442 loss_dual: 0.939 loss_total: 8.382 batch_cost: 0.20151 sec reader_cost: 0.00030 sec ips: 39.69989 images/s eta: 0:15:45
    [05/29 18:36:13] ppgan.engine.trainer INFO: Iter: 15320/20000 lr: 9.994e-05 loss_promary: 7.578 loss_dual: 1.131 loss_total: 8.709 batch_cost: 0.20125 sec reader_cost: 0.00030 sec ips: 39.75205 images/s eta: 0:15:41
    [05/29 18:36:15] ppgan.engine.trainer INFO: Iter: 15330/20000 lr: 9.994e-05 loss_promary: 9.898 loss_dual: 1.199 loss_total: 11.097 batch_cost: 0.20374 sec reader_cost: 0.00030 sec ips: 39.26616 images/s eta: 0:15:51
    [05/29 18:36:18] ppgan.engine.trainer INFO: Iter: 15340/20000 lr: 9.994e-05 loss_promary: 9.176 loss_dual: 1.126 loss_total: 10.302 batch_cost: 0.20123 sec reader_cost: 0.00029 sec ips: 39.75627 images/s eta: 0:15:37
    [05/29 18:36:20] ppgan.engine.trainer INFO: Iter: 15350/20000 lr: 9.994e-05 loss_promary: 10.471 loss_dual: 1.196 loss_total: 11.667 batch_cost: 0.19888 sec reader_cost: 0.00028 sec ips: 40.22427 images/s eta: 0:15:24
    [05/29 18:36:22] ppgan.engine.trainer INFO: Iter: 15360/20000 lr: 9.994e-05 loss_promary: 10.599 loss_dual: 1.224 loss_total: 11.823 batch_cost: 0.20112 sec reader_cost: 0.00030 sec ips: 39.77684 images/s eta: 0:15:33
    [05/29 18:36:24] ppgan.engine.trainer INFO: Iter: 15370/20000 lr: 9.994e-05 loss_promary: 7.141 loss_dual: 0.928 loss_total: 8.069 batch_cost: 0.21193 sec reader_cost: 0.00029 sec ips: 37.74891 images/s eta: 0:16:21
    [05/29 18:36:26] ppgan.engine.trainer INFO: Iter: 15380/20000 lr: 9.994e-05 loss_promary: 14.058 loss_dual: 1.437 loss_total: 15.496 batch_cost: 0.20807 sec reader_cost: 0.00030 sec ips: 38.44942 images/s eta: 0:16:01
    [05/29 18:36:28] ppgan.engine.trainer INFO: Iter: 15390/20000 lr: 9.994e-05 loss_promary: 8.256 loss_dual: 0.949 loss_total: 9.205 batch_cost: 0.19764 sec reader_cost: 0.00029 sec ips: 40.47724 images/s eta: 0:15:11
    [05/29 18:36:30] ppgan.engine.trainer INFO: Iter: 15400/20000 lr: 9.994e-05 loss_promary: 10.103 loss_dual: 1.228 loss_total: 11.330 batch_cost: 0.20069 sec reader_cost: 0.00029 sec ips: 39.86288 images/s eta: 0:15:23
    [05/29 18:36:32] ppgan.engine.trainer INFO: Iter: 15410/20000 lr: 9.994e-05 loss_promary: 8.102 loss_dual: 1.058 loss_total: 9.160 batch_cost: 0.20073 sec reader_cost: 0.00031 sec ips: 39.85527 images/s eta: 0:15:21
    [05/29 18:36:34] ppgan.engine.trainer INFO: Iter: 15420/20000 lr: 9.994e-05 loss_promary: 8.878 loss_dual: 1.065 loss_total: 9.943 batch_cost: 0.20014 sec reader_cost: 0.00031 sec ips: 39.97125 images/s eta: 0:15:16
    [05/29 18:36:36] ppgan.engine.trainer INFO: Iter: 15430/20000 lr: 9.994e-05 loss_promary: 11.578 loss_dual: 1.247 loss_total: 12.825 batch_cost: 0.20072 sec reader_cost: 0.00030 sec ips: 39.85563 images/s eta: 0:15:17
    [05/29 18:36:38] ppgan.engine.trainer INFO: Iter: 15440/20000 lr: 9.994e-05 loss_promary: 10.081 loss_dual: 1.136 loss_total: 11.217 batch_cost: 0.19972 sec reader_cost: 0.00031 sec ips: 40.05680 images/s eta: 0:15:10
    [05/29 18:36:40] ppgan.engine.trainer INFO: Iter: 15450/20000 lr: 9.994e-05 loss_promary: 9.961 loss_dual: 1.168 loss_total: 11.129 batch_cost: 0.19717 sec reader_cost: 0.00030 sec ips: 40.57452 images/s eta: 0:14:57
    [05/29 18:36:42] ppgan.engine.trainer INFO: Iter: 15460/20000 lr: 9.994e-05 loss_promary: 8.556 loss_dual: 0.964 loss_total: 9.520 batch_cost: 0.20269 sec reader_cost: 0.00032 sec ips: 39.46952 images/s eta: 0:15:20
    [05/29 18:36:44] ppgan.engine.trainer INFO: Iter: 15470/20000 lr: 9.994e-05 loss_promary: 7.135 loss_dual: 0.881 loss_total: 8.016 batch_cost: 0.20270 sec reader_cost: 0.00032 sec ips: 39.46767 images/s eta: 0:15:18
    [05/29 18:36:46] ppgan.engine.trainer INFO: Iter: 15480/20000 lr: 9.994e-05 loss_promary: 9.897 loss_dual: 1.112 loss_total: 11.010 batch_cost: 0.20330 sec reader_cost: 0.00031 sec ips: 39.35081 images/s eta: 0:15:18
    [05/29 18:36:48] ppgan.engine.trainer INFO: Iter: 15490/20000 lr: 9.994e-05 loss_promary: 13.949 loss_dual: 1.407 loss_total: 15.356 batch_cost: 0.20002 sec reader_cost: 0.00031 sec ips: 39.99573 images/s eta: 0:15:02
    [05/29 18:36:50] ppgan.engine.trainer INFO: Iter: 15500/20000 lr: 9.994e-05 loss_promary: 9.612 loss_dual: 1.048 loss_total: 10.660 batch_cost: 0.19813 sec reader_cost: 0.00030 sec ips: 40.37831 images/s eta: 0:14:51
    [05/29 18:36:52] ppgan.engine.trainer INFO: Iter: 15510/20000 lr: 9.994e-05 loss_promary: 16.394 loss_dual: 1.716 loss_total: 18.110 batch_cost: 0.20211 sec reader_cost: 0.00033 sec ips: 39.58264 images/s eta: 0:15:07
    [05/29 18:36:54] ppgan.engine.trainer INFO: Iter: 15520/20000 lr: 9.994e-05 loss_promary: 9.933 loss_dual: 1.118 loss_total: 11.051 batch_cost: 0.20946 sec reader_cost: 0.00032 sec ips: 38.19355 images/s eta: 0:15:38
    [05/29 18:36:56] ppgan.engine.trainer INFO: Iter: 15530/20000 lr: 9.994e-05 loss_promary: 6.524 loss_dual: 0.859 loss_total: 7.383 batch_cost: 0.21923 sec reader_cost: 0.00030 sec ips: 36.49183 images/s eta: 0:16:19
    [05/29 18:36:58] ppgan.engine.trainer INFO: Iter: 15540/20000 lr: 9.994e-05 loss_promary: 11.378 loss_dual: 1.245 loss_total: 12.623 batch_cost: 0.20343 sec reader_cost: 0.00030 sec ips: 39.32579 images/s eta: 0:15:07
    [05/29 18:37:00] ppgan.engine.trainer INFO: Iter: 15550/20000 lr: 9.994e-05 loss_promary: 9.866 loss_dual: 1.095 loss_total: 10.961 batch_cost: 0.20030 sec reader_cost: 0.00029 sec ips: 39.94053 images/s eta: 0:14:51
    [05/29 18:37:02] ppgan.engine.trainer INFO: Iter: 15560/20000 lr: 9.994e-05 loss_promary: 8.096 loss_dual: 0.907 loss_total: 9.003 batch_cost: 0.19962 sec reader_cost: 0.00029 sec ips: 40.07630 images/s eta: 0:14:46
    [05/29 18:37:04] ppgan.engine.trainer INFO: Iter: 15570/20000 lr: 9.994e-05 loss_promary: 7.554 loss_dual: 0.886 loss_total: 8.440 batch_cost: 0.19938 sec reader_cost: 0.00028 sec ips: 40.12352 images/s eta: 0:14:43
    [05/29 18:37:06] ppgan.engine.trainer INFO: Iter: 15580/20000 lr: 9.994e-05 loss_promary: 9.356 loss_dual: 1.142 loss_total: 10.498 batch_cost: 0.19767 sec reader_cost: 0.00027 sec ips: 40.47127 images/s eta: 0:14:33
    [05/29 18:37:08] ppgan.engine.trainer INFO: Iter: 15590/20000 lr: 9.994e-05 loss_promary: 11.203 loss_dual: 1.281 loss_total: 12.484 batch_cost: 0.19724 sec reader_cost: 0.00028 sec ips: 40.55915 images/s eta: 0:14:29
    [05/29 18:37:10] ppgan.engine.trainer INFO: Iter: 15600/20000 lr: 9.994e-05 loss_promary: 10.096 loss_dual: 1.174 loss_total: 11.270 batch_cost: 0.19808 sec reader_cost: 0.00028 sec ips: 40.38699 images/s eta: 0:14:31
    [05/29 18:37:12] ppgan.engine.trainer INFO: Iter: 15610/20000 lr: 9.994e-05 loss_promary: 10.154 loss_dual: 1.172 loss_total: 11.326 batch_cost: 0.19970 sec reader_cost: 0.00029 sec ips: 40.05965 images/s eta: 0:14:36
    [05/29 18:37:14] ppgan.engine.trainer INFO: Iter: 15620/20000 lr: 9.994e-05 loss_promary: 7.932 loss_dual: 1.060 loss_total: 8.991 batch_cost: 0.19899 sec reader_cost: 0.00028 sec ips: 40.20385 images/s eta: 0:14:31
    [05/29 18:37:16] ppgan.engine.trainer INFO: Iter: 15630/20000 lr: 9.994e-05 loss_promary: 10.446 loss_dual: 1.179 loss_total: 11.626 batch_cost: 0.19879 sec reader_cost: 0.00029 sec ips: 40.24441 images/s eta: 0:14:28
    [05/29 18:37:18] ppgan.engine.trainer INFO: Iter: 15640/20000 lr: 9.994e-05 loss_promary: 10.189 loss_dual: 1.191 loss_total: 11.379 batch_cost: 0.19495 sec reader_cost: 0.00028 sec ips: 41.03714 images/s eta: 0:14:09
    [05/29 18:37:20] ppgan.engine.trainer INFO: Iter: 15650/20000 lr: 9.994e-05 loss_promary: 10.361 loss_dual: 1.219 loss_total: 11.580 batch_cost: 0.19922 sec reader_cost: 0.00028 sec ips: 40.15755 images/s eta: 0:14:26
    [05/29 18:37:22] ppgan.engine.trainer INFO: Iter: 15660/20000 lr: 9.994e-05 loss_promary: 9.931 loss_dual: 1.077 loss_total: 11.008 batch_cost: 0.20328 sec reader_cost: 0.00030 sec ips: 39.35415 images/s eta: 0:14:42
    [05/29 18:37:24] ppgan.engine.trainer INFO: Iter: 15670/20000 lr: 9.994e-05 loss_promary: 13.018 loss_dual: 1.412 loss_total: 14.430 batch_cost: 0.20153 sec reader_cost: 0.00030 sec ips: 39.69647 images/s eta: 0:14:32
    [05/29 18:37:26] ppgan.engine.trainer INFO: Iter: 15680/20000 lr: 9.994e-05 loss_promary: 8.516 loss_dual: 0.930 loss_total: 9.446 batch_cost: 0.21320 sec reader_cost: 0.00029 sec ips: 37.52296 images/s eta: 0:15:21
    [05/29 18:37:28] ppgan.engine.trainer INFO: Iter: 15690/20000 lr: 9.994e-05 loss_promary: 7.120 loss_dual: 0.846 loss_total: 7.965 batch_cost: 0.20773 sec reader_cost: 0.00029 sec ips: 38.51151 images/s eta: 0:14:55
    [05/29 18:37:30] ppgan.engine.trainer INFO: Iter: 15700/20000 lr: 9.994e-05 loss_promary: 8.664 loss_dual: 1.049 loss_total: 9.713 batch_cost: 0.20052 sec reader_cost: 0.00030 sec ips: 39.89639 images/s eta: 0:14:22
    [05/29 18:37:32] ppgan.engine.trainer INFO: Iter: 15710/20000 lr: 9.994e-05 loss_promary: 10.184 loss_dual: 1.199 loss_total: 11.383 batch_cost: 0.20436 sec reader_cost: 0.00029 sec ips: 39.14711 images/s eta: 0:14:36
    [05/29 18:37:34] ppgan.engine.trainer INFO: Iter: 15720/20000 lr: 9.994e-05 loss_promary: 12.804 loss_dual: 1.424 loss_total: 14.228 batch_cost: 0.19951 sec reader_cost: 0.00029 sec ips: 40.09836 images/s eta: 0:14:13
    [05/29 18:37:36] ppgan.engine.trainer INFO: Iter: 15730/20000 lr: 9.994e-05 loss_promary: 10.012 loss_dual: 1.180 loss_total: 11.192 batch_cost: 0.19902 sec reader_cost: 0.00029 sec ips: 40.19795 images/s eta: 0:14:09
    [05/29 18:37:38] ppgan.engine.trainer INFO: Iter: 15740/20000 lr: 9.994e-05 loss_promary: 11.451 loss_dual: 1.255 loss_total: 12.706 batch_cost: 0.19873 sec reader_cost: 0.00028 sec ips: 40.25503 images/s eta: 0:14:06
    [05/29 18:37:40] ppgan.engine.trainer INFO: Iter: 15750/20000 lr: 9.994e-05 loss_promary: 11.404 loss_dual: 1.314 loss_total: 12.718 batch_cost: 0.19740 sec reader_cost: 0.00028 sec ips: 40.52659 images/s eta: 0:13:58
    [05/29 18:37:42] ppgan.engine.trainer INFO: Iter: 15760/20000 lr: 9.994e-05 loss_promary: 9.914 loss_dual: 1.060 loss_total: 10.974 batch_cost: 0.19916 sec reader_cost: 0.00029 sec ips: 40.16870 images/s eta: 0:14:04
    [05/29 18:37:44] ppgan.engine.trainer INFO: Iter: 15770/20000 lr: 9.994e-05 loss_promary: 12.028 loss_dual: 1.351 loss_total: 13.379 batch_cost: 0.20363 sec reader_cost: 0.00028 sec ips: 39.28692 images/s eta: 0:14:21
    [05/29 18:37:46] ppgan.engine.trainer INFO: Iter: 15780/20000 lr: 9.994e-05 loss_promary: 6.618 loss_dual: 0.854 loss_total: 7.472 batch_cost: 0.20245 sec reader_cost: 0.00030 sec ips: 39.51592 images/s eta: 0:14:14
    [05/29 18:37:48] ppgan.engine.trainer INFO: Iter: 15790/20000 lr: 9.994e-05 loss_promary: 9.478 loss_dual: 1.124 loss_total: 10.601 batch_cost: 0.19963 sec reader_cost: 0.00029 sec ips: 40.07429 images/s eta: 0:14:00
    [05/29 18:37:50] ppgan.engine.trainer INFO: Iter: 15800/20000 lr: 9.994e-05 loss_promary: 11.515 loss_dual: 1.206 loss_total: 12.721 batch_cost: 0.20289 sec reader_cost: 0.00030 sec ips: 39.43014 images/s eta: 0:14:12
    [05/29 18:37:52] ppgan.engine.trainer INFO: Iter: 15810/20000 lr: 9.994e-05 loss_promary: 9.073 loss_dual: 1.007 loss_total: 10.080 batch_cost: 0.20647 sec reader_cost: 0.00030 sec ips: 38.74703 images/s eta: 0:14:25
    [05/29 18:37:55] ppgan.engine.trainer INFO: Iter: 15820/20000 lr: 9.994e-05 loss_promary: 13.683 loss_dual: 1.419 loss_total: 15.102 batch_cost: 0.20601 sec reader_cost: 0.00030 sec ips: 38.83338 images/s eta: 0:14:21
    [05/29 18:37:57] ppgan.engine.trainer INFO: Iter: 15830/20000 lr: 9.994e-05 loss_promary: 10.476 loss_dual: 1.109 loss_total: 11.585 batch_cost: 0.20797 sec reader_cost: 0.00030 sec ips: 38.46769 images/s eta: 0:14:27
    [05/29 18:37:59] ppgan.engine.trainer INFO: Iter: 15840/20000 lr: 9.994e-05 loss_promary: 10.258 loss_dual: 1.108 loss_total: 11.366 batch_cost: 0.23329 sec reader_cost: 0.00031 sec ips: 34.29224 images/s eta: 0:16:10
    [05/29 18:38:01] ppgan.engine.trainer INFO: Iter: 15850/20000 lr: 9.994e-05 loss_promary: 8.814 loss_dual: 0.979 loss_total: 9.794 batch_cost: 0.22260 sec reader_cost: 0.00030 sec ips: 35.93873 images/s eta: 0:15:23
    [05/29 18:38:03] ppgan.engine.trainer INFO: Iter: 15860/20000 lr: 9.994e-05 loss_promary: 10.530 loss_dual: 1.166 loss_total: 11.696 batch_cost: 0.20402 sec reader_cost: 0.00029 sec ips: 39.21207 images/s eta: 0:14:04
    [05/29 18:38:05] ppgan.engine.trainer INFO: Iter: 15870/20000 lr: 9.994e-05 loss_promary: 12.588 loss_dual: 1.405 loss_total: 13.993 batch_cost: 0.20322 sec reader_cost: 0.00029 sec ips: 39.36605 images/s eta: 0:13:59
    [05/29 18:38:07] ppgan.engine.trainer INFO: Iter: 15880/20000 lr: 9.994e-05 loss_promary: 9.092 loss_dual: 1.046 loss_total: 10.138 batch_cost: 0.20392 sec reader_cost: 0.00030 sec ips: 39.23167 images/s eta: 0:14:00
    [05/29 18:38:09] ppgan.engine.trainer INFO: Iter: 15890/20000 lr: 9.994e-05 loss_promary: 9.670 loss_dual: 1.059 loss_total: 10.730 batch_cost: 0.20389 sec reader_cost: 0.00030 sec ips: 39.23756 images/s eta: 0:13:57
    [05/29 18:38:11] ppgan.engine.trainer INFO: Iter: 15900/20000 lr: 9.994e-05 loss_promary: 8.808 loss_dual: 1.030 loss_total: 9.838 batch_cost: 0.20262 sec reader_cost: 0.00030 sec ips: 39.48191 images/s eta: 0:13:50
    [05/29 18:38:13] ppgan.engine.trainer INFO: Iter: 15910/20000 lr: 9.994e-05 loss_promary: 7.417 loss_dual: 1.024 loss_total: 8.441 batch_cost: 0.20143 sec reader_cost: 0.00029 sec ips: 39.71596 images/s eta: 0:13:43
    [05/29 18:38:15] ppgan.engine.trainer INFO: Iter: 15920/20000 lr: 9.994e-05 loss_promary: 9.327 loss_dual: 1.070 loss_total: 10.397 batch_cost: 0.20849 sec reader_cost: 0.00030 sec ips: 38.37174 images/s eta: 0:14:10
    [05/29 18:38:17] ppgan.engine.trainer INFO: Iter: 15930/20000 lr: 9.994e-05 loss_promary: 8.953 loss_dual: 1.053 loss_total: 10.006 batch_cost: 0.19982 sec reader_cost: 0.00029 sec ips: 40.03686 images/s eta: 0:13:33
    [05/29 18:38:20] ppgan.engine.trainer INFO: Iter: 15940/20000 lr: 9.994e-05 loss_promary: 7.090 loss_dual: 0.948 loss_total: 8.038 batch_cost: 0.20319 sec reader_cost: 0.00029 sec ips: 39.37133 images/s eta: 0:13:44
    [05/29 18:38:22] ppgan.engine.trainer INFO: Iter: 15950/20000 lr: 9.994e-05 loss_promary: 8.661 loss_dual: 0.998 loss_total: 9.660 batch_cost: 0.20661 sec reader_cost: 0.00031 sec ips: 38.72091 images/s eta: 0:13:56
    [05/29 18:38:24] ppgan.engine.trainer INFO: Iter: 15960/20000 lr: 9.994e-05 loss_promary: 8.200 loss_dual: 1.071 loss_total: 9.271 batch_cost: 0.20356 sec reader_cost: 0.00039 sec ips: 39.30086 images/s eta: 0:13:42
    [05/29 18:38:26] ppgan.engine.trainer INFO: Iter: 15970/20000 lr: 9.994e-05 loss_promary: 10.578 loss_dual: 1.413 loss_total: 11.990 batch_cost: 0.26087 sec reader_cost: 0.04462 sec ips: 30.66620 images/s eta: 0:17:31
    [05/29 18:38:28] ppgan.engine.trainer INFO: Iter: 15980/20000 lr: 9.994e-05 loss_promary: 8.218 loss_dual: 0.898 loss_total: 9.116 batch_cost: 0.20886 sec reader_cost: 0.00055 sec ips: 38.30330 images/s eta: 0:13:59
    [05/29 18:38:30] ppgan.engine.trainer INFO: Iter: 15990/20000 lr: 9.994e-05 loss_promary: 7.171 loss_dual: 0.885 loss_total: 8.056 batch_cost: 0.21299 sec reader_cost: 0.00029 sec ips: 37.56030 images/s eta: 0:14:14
    [05/29 18:38:33] ppgan.engine.trainer INFO: Iter: 16000/20000 lr: 9.994e-05 loss_promary: 11.433 loss_dual: 1.208 loss_total: 12.641 batch_cost: 0.21825 sec reader_cost: 0.00030 sec ips: 36.65523 images/s eta: 0:14:32
    [05/29 18:38:33] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:38:34] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:38:36] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:38:38] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:38:40] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:38:41] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:38:43] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:38:45] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:38:46] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:38:48] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:38:50] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:38:52] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:38:53] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:38:55] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:38:57] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:38:58] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:39:00] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:39:02] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:39:04] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:39:05] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:39:07] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:39:09] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:39:11] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:39:12] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:39:14] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:39:16] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:39:17] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:39:19] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:39:21] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:39:22] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:39:24] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:39:26] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:39:27] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:39:29] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:39:31] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:39:33] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:39:34] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:39:36] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:39:38] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:39:39] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:39:41] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:39:43] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:39:45] ppgan.engine.trainer INFO: Metric psnr: 29.4084
    [05/29 18:39:45] ppgan.engine.trainer INFO: Metric ssim: 0.8004
    [05/29 18:39:47] ppgan.engine.trainer INFO: Iter: 16010/20000 lr: 9.994e-05 loss_promary: 10.232 loss_dual: 1.125 loss_total: 11.358 batch_cost: 0.20714 sec reader_cost: 0.00036 sec ips: 38.62168 images/s eta: 0:13:46
    [05/29 18:39:49] ppgan.engine.trainer INFO: Iter: 16020/20000 lr: 9.994e-05 loss_promary: 7.190 loss_dual: 0.863 loss_total: 8.053 batch_cost: 0.20686 sec reader_cost: 0.00032 sec ips: 38.67400 images/s eta: 0:13:43
    [05/29 18:39:51] ppgan.engine.trainer INFO: Iter: 16030/20000 lr: 9.994e-05 loss_promary: 9.918 loss_dual: 1.168 loss_total: 11.086 batch_cost: 0.20903 sec reader_cost: 0.00029 sec ips: 38.27188 images/s eta: 0:13:49
    [05/29 18:39:53] ppgan.engine.trainer INFO: Iter: 16040/20000 lr: 9.994e-05 loss_promary: 12.174 loss_dual: 1.276 loss_total: 13.450 batch_cost: 0.20925 sec reader_cost: 0.00030 sec ips: 38.23182 images/s eta: 0:13:48
    [05/29 18:39:55] ppgan.engine.trainer INFO: Iter: 16050/20000 lr: 9.994e-05 loss_promary: 11.471 loss_dual: 1.237 loss_total: 12.708 batch_cost: 0.21364 sec reader_cost: 0.00035 sec ips: 37.44663 images/s eta: 0:14:03
    [05/29 18:39:58] ppgan.engine.trainer INFO: Iter: 16060/20000 lr: 9.994e-05 loss_promary: 11.503 loss_dual: 1.259 loss_total: 12.761 batch_cost: 0.21307 sec reader_cost: 0.00031 sec ips: 37.54555 images/s eta: 0:13:59
    [05/29 18:40:00] ppgan.engine.trainer INFO: Iter: 16070/20000 lr: 9.994e-05 loss_promary: 8.111 loss_dual: 0.968 loss_total: 9.079 batch_cost: 0.25102 sec reader_cost: 0.00031 sec ips: 31.86996 images/s eta: 0:16:26
    [05/29 18:40:02] ppgan.engine.trainer INFO: Iter: 16080/20000 lr: 9.994e-05 loss_promary: 8.316 loss_dual: 1.011 loss_total: 9.327 batch_cost: 0.20696 sec reader_cost: 0.00030 sec ips: 38.65429 images/s eta: 0:13:31
    [05/29 18:40:04] ppgan.engine.trainer INFO: Iter: 16090/20000 lr: 9.994e-05 loss_promary: 12.033 loss_dual: 1.266 loss_total: 13.299 batch_cost: 0.20938 sec reader_cost: 0.00032 sec ips: 38.20833 images/s eta: 0:13:38
    [05/29 18:40:07] ppgan.engine.trainer INFO: Iter: 16100/20000 lr: 9.994e-05 loss_promary: 8.212 loss_dual: 0.921 loss_total: 9.133 batch_cost: 0.23553 sec reader_cost: 0.00073 sec ips: 33.96527 images/s eta: 0:15:18
    [05/29 18:40:09] ppgan.engine.trainer INFO: Iter: 16110/20000 lr: 9.994e-05 loss_promary: 11.076 loss_dual: 1.166 loss_total: 12.242 batch_cost: 0.20921 sec reader_cost: 0.00033 sec ips: 38.23951 images/s eta: 0:13:33
    [05/29 18:40:11] ppgan.engine.trainer INFO: Iter: 16120/20000 lr: 9.994e-05 loss_promary: 10.588 loss_dual: 1.190 loss_total: 11.779 batch_cost: 0.21435 sec reader_cost: 0.00032 sec ips: 37.32279 images/s eta: 0:13:51
    [05/29 18:40:13] ppgan.engine.trainer INFO: Iter: 16130/20000 lr: 9.994e-05 loss_promary: 9.272 loss_dual: 1.065 loss_total: 10.336 batch_cost: 0.21171 sec reader_cost: 0.00032 sec ips: 37.78745 images/s eta: 0:13:39
    [05/29 18:40:15] ppgan.engine.trainer INFO: Iter: 16140/20000 lr: 9.994e-05 loss_promary: 11.961 loss_dual: 1.359 loss_total: 13.320 batch_cost: 0.23284 sec reader_cost: 0.00192 sec ips: 34.35769 images/s eta: 0:14:58
    [05/29 18:40:17] ppgan.engine.trainer INFO: Iter: 16150/20000 lr: 9.994e-05 loss_promary: 9.937 loss_dual: 1.135 loss_total: 11.072 batch_cost: 0.21944 sec reader_cost: 0.00032 sec ips: 36.45695 images/s eta: 0:14:04
    [05/29 18:40:20] ppgan.engine.trainer INFO: Iter: 16160/20000 lr: 9.994e-05 loss_promary: 10.096 loss_dual: 1.165 loss_total: 11.260 batch_cost: 0.21746 sec reader_cost: 0.00031 sec ips: 36.78760 images/s eta: 0:13:55
    [05/29 18:40:22] ppgan.engine.trainer INFO: Iter: 16170/20000 lr: 9.994e-05 loss_promary: 7.396 loss_dual: 0.887 loss_total: 8.282 batch_cost: 0.21268 sec reader_cost: 0.00030 sec ips: 37.61511 images/s eta: 0:13:34
    [05/29 18:40:24] ppgan.engine.trainer INFO: Iter: 16180/20000 lr: 9.994e-05 loss_promary: 7.823 loss_dual: 0.948 loss_total: 8.771 batch_cost: 0.20749 sec reader_cost: 0.00032 sec ips: 38.55635 images/s eta: 0:13:12
    [05/29 18:40:26] ppgan.engine.trainer INFO: Iter: 16190/20000 lr: 9.994e-05 loss_promary: 8.147 loss_dual: 0.924 loss_total: 9.072 batch_cost: 0.20493 sec reader_cost: 0.00031 sec ips: 39.03834 images/s eta: 0:13:00
    [05/29 18:40:28] ppgan.engine.trainer INFO: Iter: 16200/20000 lr: 9.994e-05 loss_promary: 9.744 loss_dual: 1.140 loss_total: 10.884 batch_cost: 0.20603 sec reader_cost: 0.00032 sec ips: 38.82963 images/s eta: 0:13:02
    [05/29 18:40:30] ppgan.engine.trainer INFO: Iter: 16210/20000 lr: 9.994e-05 loss_promary: 8.482 loss_dual: 0.973 loss_total: 9.455 batch_cost: 0.24946 sec reader_cost: 0.00032 sec ips: 32.06928 images/s eta: 0:15:45
    [05/29 18:40:33] ppgan.engine.trainer INFO: Iter: 16220/20000 lr: 9.994e-05 loss_promary: 12.025 loss_dual: 1.293 loss_total: 13.318 batch_cost: 0.20883 sec reader_cost: 0.00031 sec ips: 38.30804 images/s eta: 0:13:09
    [05/29 18:40:35] ppgan.engine.trainer INFO: Iter: 16230/20000 lr: 9.994e-05 loss_promary: 7.177 loss_dual: 0.952 loss_total: 8.129 batch_cost: 0.20564 sec reader_cost: 0.00030 sec ips: 38.90290 images/s eta: 0:12:55
    [05/29 18:40:37] ppgan.engine.trainer INFO: Iter: 16240/20000 lr: 9.994e-05 loss_promary: 12.048 loss_dual: 1.382 loss_total: 13.429 batch_cost: 0.21137 sec reader_cost: 0.00032 sec ips: 37.84869 images/s eta: 0:13:14
    [05/29 18:40:39] ppgan.engine.trainer INFO: Iter: 16250/20000 lr: 9.993e-05 loss_promary: 7.444 loss_dual: 0.919 loss_total: 8.363 batch_cost: 0.22411 sec reader_cost: 0.00031 sec ips: 35.69596 images/s eta: 0:14:00
    [05/29 18:40:41] ppgan.engine.trainer INFO: Iter: 16260/20000 lr: 9.993e-05 loss_promary: 8.628 loss_dual: 1.169 loss_total: 9.797 batch_cost: 0.20933 sec reader_cost: 0.00029 sec ips: 38.21748 images/s eta: 0:13:02
    [05/29 18:40:43] ppgan.engine.trainer INFO: Iter: 16270/20000 lr: 9.993e-05 loss_promary: 10.378 loss_dual: 1.183 loss_total: 11.561 batch_cost: 0.21227 sec reader_cost: 0.00030 sec ips: 37.68794 images/s eta: 0:13:11
    [05/29 18:40:45] ppgan.engine.trainer INFO: Iter: 16280/20000 lr: 9.993e-05 loss_promary: 12.492 loss_dual: 1.367 loss_total: 13.859 batch_cost: 0.21101 sec reader_cost: 0.00031 sec ips: 37.91236 images/s eta: 0:13:04
    [05/29 18:40:47] ppgan.engine.trainer INFO: Iter: 16290/20000 lr: 9.993e-05 loss_promary: 8.314 loss_dual: 1.009 loss_total: 9.322 batch_cost: 0.21194 sec reader_cost: 0.00030 sec ips: 37.74668 images/s eta: 0:13:06
    [05/29 18:40:49] ppgan.engine.trainer INFO: Iter: 16300/20000 lr: 9.993e-05 loss_promary: 13.138 loss_dual: 1.454 loss_total: 14.592 batch_cost: 0.20685 sec reader_cost: 0.00030 sec ips: 38.67455 images/s eta: 0:12:45
    [05/29 18:40:52] ppgan.engine.trainer INFO: Iter: 16310/20000 lr: 9.993e-05 loss_promary: 8.605 loss_dual: 0.990 loss_total: 9.595 batch_cost: 0.20766 sec reader_cost: 0.00029 sec ips: 38.52416 images/s eta: 0:12:46
    [05/29 18:40:54] ppgan.engine.trainer INFO: Iter: 16320/20000 lr: 9.993e-05 loss_promary: 6.816 loss_dual: 0.825 loss_total: 7.640 batch_cost: 0.21311 sec reader_cost: 0.00032 sec ips: 37.53865 images/s eta: 0:13:04
    [05/29 18:40:56] ppgan.engine.trainer INFO: Iter: 16330/20000 lr: 9.993e-05 loss_promary: 6.724 loss_dual: 0.821 loss_total: 7.545 batch_cost: 0.21269 sec reader_cost: 0.00031 sec ips: 37.61396 images/s eta: 0:13:00
    [05/29 18:40:58] ppgan.engine.trainer INFO: Iter: 16340/20000 lr: 9.993e-05 loss_promary: 10.736 loss_dual: 1.282 loss_total: 12.018 batch_cost: 0.20794 sec reader_cost: 0.00031 sec ips: 38.47215 images/s eta: 0:12:41
    [05/29 18:41:00] ppgan.engine.trainer INFO: Iter: 16350/20000 lr: 9.993e-05 loss_promary: 8.684 loss_dual: 1.022 loss_total: 9.706 batch_cost: 0.22707 sec reader_cost: 0.00031 sec ips: 35.23115 images/s eta: 0:13:48
    [05/29 18:41:02] ppgan.engine.trainer INFO: Iter: 16360/20000 lr: 9.993e-05 loss_promary: 7.599 loss_dual: 0.908 loss_total: 8.507 batch_cost: 0.20477 sec reader_cost: 0.00029 sec ips: 39.06916 images/s eta: 0:12:25
    [05/29 18:41:04] ppgan.engine.trainer INFO: Iter: 16370/20000 lr: 9.993e-05 loss_promary: 8.564 loss_dual: 0.978 loss_total: 9.542 batch_cost: 0.20465 sec reader_cost: 0.00031 sec ips: 39.09205 images/s eta: 0:12:22
    [05/29 18:41:06] ppgan.engine.trainer INFO: Iter: 16380/20000 lr: 9.993e-05 loss_promary: 15.246 loss_dual: 1.536 loss_total: 16.782 batch_cost: 0.20860 sec reader_cost: 0.00033 sec ips: 38.35157 images/s eta: 0:12:35
    [05/29 18:41:08] ppgan.engine.trainer INFO: Iter: 16390/20000 lr: 9.993e-05 loss_promary: 7.151 loss_dual: 0.913 loss_total: 8.064 batch_cost: 0.20418 sec reader_cost: 0.00031 sec ips: 39.18156 images/s eta: 0:12:17
    [05/29 18:41:11] ppgan.engine.trainer INFO: Iter: 16400/20000 lr: 9.993e-05 loss_promary: 8.340 loss_dual: 1.008 loss_total: 9.348 batch_cost: 0.22635 sec reader_cost: 0.00031 sec ips: 35.34321 images/s eta: 0:13:34
    [05/29 18:41:13] ppgan.engine.trainer INFO: Iter: 16410/20000 lr: 9.993e-05 loss_promary: 8.579 loss_dual: 0.925 loss_total: 9.504 batch_cost: 0.21244 sec reader_cost: 0.00032 sec ips: 37.65814 images/s eta: 0:12:42
    [05/29 18:41:15] ppgan.engine.trainer INFO: Iter: 16420/20000 lr: 9.993e-05 loss_promary: 11.447 loss_dual: 1.258 loss_total: 12.705 batch_cost: 0.21559 sec reader_cost: 0.00032 sec ips: 37.10742 images/s eta: 0:12:51
    [05/29 18:41:17] ppgan.engine.trainer INFO: Iter: 16430/20000 lr: 9.993e-05 loss_promary: 9.902 loss_dual: 1.035 loss_total: 10.937 batch_cost: 0.20662 sec reader_cost: 0.00032 sec ips: 38.71911 images/s eta: 0:12:17
    [05/29 18:41:19] ppgan.engine.trainer INFO: Iter: 16440/20000 lr: 9.993e-05 loss_promary: 8.560 loss_dual: 1.083 loss_total: 9.642 batch_cost: 0.20141 sec reader_cost: 0.00031 sec ips: 39.72047 images/s eta: 0:11:57
    [05/29 18:41:21] ppgan.engine.trainer INFO: Iter: 16450/20000 lr: 9.993e-05 loss_promary: 9.563 loss_dual: 1.118 loss_total: 10.681 batch_cost: 0.20389 sec reader_cost: 0.00032 sec ips: 39.23752 images/s eta: 0:12:03
    [05/29 18:41:23] ppgan.engine.trainer INFO: Iter: 16460/20000 lr: 9.993e-05 loss_promary: 9.574 loss_dual: 1.129 loss_total: 10.704 batch_cost: 0.20203 sec reader_cost: 0.00032 sec ips: 39.59757 images/s eta: 0:11:55
    [05/29 18:41:25] ppgan.engine.trainer INFO: Iter: 16470/20000 lr: 9.993e-05 loss_promary: 9.825 loss_dual: 1.119 loss_total: 10.943 batch_cost: 0.20238 sec reader_cost: 0.00032 sec ips: 39.53032 images/s eta: 0:11:54
    [05/29 18:41:27] ppgan.engine.trainer INFO: Iter: 16480/20000 lr: 9.993e-05 loss_promary: 8.333 loss_dual: 0.957 loss_total: 9.291 batch_cost: 0.20473 sec reader_cost: 0.00031 sec ips: 39.07588 images/s eta: 0:12:00
    [05/29 18:41:29] ppgan.engine.trainer INFO: Iter: 16490/20000 lr: 9.993e-05 loss_promary: 7.952 loss_dual: 0.944 loss_total: 8.896 batch_cost: 0.20323 sec reader_cost: 0.00031 sec ips: 39.36436 images/s eta: 0:11:53
    [05/29 18:41:31] ppgan.engine.trainer INFO: Iter: 16500/20000 lr: 9.993e-05 loss_promary: 8.057 loss_dual: 0.967 loss_total: 9.024 batch_cost: 0.21078 sec reader_cost: 0.00031 sec ips: 37.95477 images/s eta: 0:12:17
    [05/29 18:41:33] ppgan.engine.trainer INFO: Iter: 16510/20000 lr: 9.993e-05 loss_promary: 11.023 loss_dual: 1.124 loss_total: 12.148 batch_cost: 0.20506 sec reader_cost: 0.00034 sec ips: 39.01222 images/s eta: 0:11:55
    [05/29 18:41:35] ppgan.engine.trainer INFO: Iter: 16520/20000 lr: 9.993e-05 loss_promary: 10.185 loss_dual: 1.144 loss_total: 11.328 batch_cost: 0.20276 sec reader_cost: 0.00030 sec ips: 39.45492 images/s eta: 0:11:45
    [05/29 18:41:37] ppgan.engine.trainer INFO: Iter: 16530/20000 lr: 9.993e-05 loss_promary: 6.920 loss_dual: 0.803 loss_total: 7.723 batch_cost: 0.20310 sec reader_cost: 0.00030 sec ips: 39.38879 images/s eta: 0:11:44
    [05/29 18:41:40] ppgan.engine.trainer INFO: Iter: 16540/20000 lr: 9.993e-05 loss_promary: 12.228 loss_dual: 1.366 loss_total: 13.594 batch_cost: 0.20122 sec reader_cost: 0.00029 sec ips: 39.75799 images/s eta: 0:11:36
    [05/29 18:41:42] ppgan.engine.trainer INFO: Iter: 16550/20000 lr: 9.993e-05 loss_promary: 10.700 loss_dual: 1.210 loss_total: 11.910 batch_cost: 0.21575 sec reader_cost: 0.00030 sec ips: 37.07976 images/s eta: 0:12:24
    [05/29 18:41:44] ppgan.engine.trainer INFO: Iter: 16560/20000 lr: 9.993e-05 loss_promary: 10.439 loss_dual: 1.154 loss_total: 11.593 batch_cost: 0.20668 sec reader_cost: 0.00030 sec ips: 38.70727 images/s eta: 0:11:50
    [05/29 18:41:46] ppgan.engine.trainer INFO: Iter: 16570/20000 lr: 9.993e-05 loss_promary: 8.818 loss_dual: 1.039 loss_total: 9.857 batch_cost: 0.20754 sec reader_cost: 0.00030 sec ips: 38.54726 images/s eta: 0:11:51
    [05/29 18:41:48] ppgan.engine.trainer INFO: Iter: 16580/20000 lr: 9.993e-05 loss_promary: 8.798 loss_dual: 0.946 loss_total: 9.744 batch_cost: 0.20398 sec reader_cost: 0.00030 sec ips: 39.21950 images/s eta: 0:11:37
    [05/29 18:41:50] ppgan.engine.trainer INFO: Iter: 16590/20000 lr: 9.993e-05 loss_promary: 8.904 loss_dual: 1.122 loss_total: 10.027 batch_cost: 0.20451 sec reader_cost: 0.00030 sec ips: 39.11779 images/s eta: 0:11:37
    [05/29 18:41:52] ppgan.engine.trainer INFO: Iter: 16600/20000 lr: 9.993e-05 loss_promary: 5.940 loss_dual: 0.754 loss_total: 6.694 batch_cost: 0.20448 sec reader_cost: 0.00030 sec ips: 39.12413 images/s eta: 0:11:35
    [05/29 18:41:54] ppgan.engine.trainer INFO: Iter: 16610/20000 lr: 9.993e-05 loss_promary: 8.433 loss_dual: 1.009 loss_total: 9.442 batch_cost: 0.20568 sec reader_cost: 0.00031 sec ips: 38.89559 images/s eta: 0:11:37
    [05/29 18:41:56] ppgan.engine.trainer INFO: Iter: 16620/20000 lr: 9.993e-05 loss_promary: 9.579 loss_dual: 1.342 loss_total: 10.921 batch_cost: 0.20532 sec reader_cost: 0.00030 sec ips: 38.96427 images/s eta: 0:11:33
    [05/29 18:41:58] ppgan.engine.trainer INFO: Iter: 16630/20000 lr: 9.993e-05 loss_promary: 10.251 loss_dual: 1.298 loss_total: 11.550 batch_cost: 0.20375 sec reader_cost: 0.00030 sec ips: 39.26442 images/s eta: 0:11:26
    [05/29 18:42:00] ppgan.engine.trainer INFO: Iter: 16640/20000 lr: 9.993e-05 loss_promary: 8.104 loss_dual: 1.051 loss_total: 9.155 batch_cost: 0.20350 sec reader_cost: 0.00030 sec ips: 39.31236 images/s eta: 0:11:23
    [05/29 18:42:02] ppgan.engine.trainer INFO: Iter: 16650/20000 lr: 9.993e-05 loss_promary: 7.685 loss_dual: 1.051 loss_total: 8.736 batch_cost: 0.20294 sec reader_cost: 0.00029 sec ips: 39.41955 images/s eta: 0:11:19
    [05/29 18:42:04] ppgan.engine.trainer INFO: Iter: 16660/20000 lr: 9.993e-05 loss_promary: 7.567 loss_dual: 0.957 loss_total: 8.523 batch_cost: 0.20343 sec reader_cost: 0.00031 sec ips: 39.32568 images/s eta: 0:11:19
    [05/29 18:42:06] ppgan.engine.trainer INFO: Iter: 16670/20000 lr: 9.993e-05 loss_promary: 9.431 loss_dual: 1.151 loss_total: 10.582 batch_cost: 0.20111 sec reader_cost: 0.00029 sec ips: 39.77926 images/s eta: 0:11:09
    [05/29 18:42:08] ppgan.engine.trainer INFO: Iter: 16680/20000 lr: 9.993e-05 loss_promary: 7.993 loss_dual: 1.047 loss_total: 9.040 batch_cost: 0.20016 sec reader_cost: 0.00029 sec ips: 39.96815 images/s eta: 0:11:04
    [05/29 18:42:10] ppgan.engine.trainer INFO: Iter: 16690/20000 lr: 9.993e-05 loss_promary: 7.180 loss_dual: 0.840 loss_total: 8.020 batch_cost: 0.20080 sec reader_cost: 0.00029 sec ips: 39.84125 images/s eta: 0:11:04
    [05/29 18:42:12] ppgan.engine.trainer INFO: Iter: 16700/20000 lr: 9.993e-05 loss_promary: 11.544 loss_dual: 1.235 loss_total: 12.779 batch_cost: 0.20467 sec reader_cost: 0.00029 sec ips: 39.08796 images/s eta: 0:11:15
    [05/29 18:42:14] ppgan.engine.trainer INFO: Iter: 16710/20000 lr: 9.993e-05 loss_promary: 9.593 loss_dual: 1.112 loss_total: 10.704 batch_cost: 0.21346 sec reader_cost: 0.00029 sec ips: 37.47709 images/s eta: 0:11:42
    [05/29 18:42:16] ppgan.engine.trainer INFO: Iter: 16720/20000 lr: 9.993e-05 loss_promary: 10.928 loss_dual: 1.171 loss_total: 12.098 batch_cost: 0.20083 sec reader_cost: 0.00029 sec ips: 39.83494 images/s eta: 0:10:58
    [05/29 18:42:19] ppgan.engine.trainer INFO: Iter: 16730/20000 lr: 9.993e-05 loss_promary: 10.233 loss_dual: 1.154 loss_total: 11.388 batch_cost: 0.21070 sec reader_cost: 0.00031 sec ips: 37.96839 images/s eta: 0:11:28
    [05/29 18:42:21] ppgan.engine.trainer INFO: Iter: 16740/20000 lr: 9.993e-05 loss_promary: 9.110 loss_dual: 1.017 loss_total: 10.127 batch_cost: 0.20729 sec reader_cost: 0.00031 sec ips: 38.59246 images/s eta: 0:11:15
    [05/29 18:42:23] ppgan.engine.trainer INFO: Iter: 16750/20000 lr: 9.993e-05 loss_promary: 5.959 loss_dual: 0.812 loss_total: 6.771 batch_cost: 0.20967 sec reader_cost: 0.00032 sec ips: 38.15549 images/s eta: 0:11:21
    [05/29 18:42:25] ppgan.engine.trainer INFO: Iter: 16760/20000 lr: 9.993e-05 loss_promary: 8.800 loss_dual: 1.043 loss_total: 9.843 batch_cost: 0.20422 sec reader_cost: 0.00031 sec ips: 39.17339 images/s eta: 0:11:01
    [05/29 18:42:27] ppgan.engine.trainer INFO: Iter: 16770/20000 lr: 9.993e-05 loss_promary: 8.889 loss_dual: 0.999 loss_total: 9.888 batch_cost: 0.20691 sec reader_cost: 0.00030 sec ips: 38.66377 images/s eta: 0:11:08
    [05/29 18:42:29] ppgan.engine.trainer INFO: Iter: 16780/20000 lr: 9.993e-05 loss_promary: 9.884 loss_dual: 1.091 loss_total: 10.974 batch_cost: 0.20444 sec reader_cost: 0.00029 sec ips: 39.13193 images/s eta: 0:10:58
    [05/29 18:42:31] ppgan.engine.trainer INFO: Iter: 16790/20000 lr: 9.993e-05 loss_promary: 10.524 loss_dual: 1.137 loss_total: 11.661 batch_cost: 0.20221 sec reader_cost: 0.00030 sec ips: 39.56292 images/s eta: 0:10:49
    [05/29 18:42:33] ppgan.engine.trainer INFO: Iter: 16800/20000 lr: 9.993e-05 loss_promary: 6.650 loss_dual: 0.850 loss_total: 7.500 batch_cost: 0.19936 sec reader_cost: 0.00036 sec ips: 40.12750 images/s eta: 0:10:37
    [05/29 18:42:36] ppgan.engine.trainer INFO: Iter: 16810/20000 lr: 9.993e-05 loss_promary: 11.440 loss_dual: 1.171 loss_total: 12.611 batch_cost: 0.26701 sec reader_cost: 0.03738 sec ips: 29.96169 images/s eta: 0:14:11
    [05/29 18:42:38] ppgan.engine.trainer INFO: Iter: 16820/20000 lr: 9.993e-05 loss_promary: 6.988 loss_dual: 0.796 loss_total: 7.784 batch_cost: 0.23017 sec reader_cost: 0.00032 sec ips: 34.75691 images/s eta: 0:12:11
    [05/29 18:42:40] ppgan.engine.trainer INFO: Iter: 16830/20000 lr: 9.993e-05 loss_promary: 10.270 loss_dual: 1.044 loss_total: 11.315 batch_cost: 0.22633 sec reader_cost: 0.00031 sec ips: 35.34714 images/s eta: 0:11:57
    [05/29 18:42:42] ppgan.engine.trainer INFO: Iter: 16840/20000 lr: 9.993e-05 loss_promary: 9.003 loss_dual: 1.040 loss_total: 10.043 batch_cost: 0.22980 sec reader_cost: 0.00032 sec ips: 34.81268 images/s eta: 0:12:06
    [05/29 18:42:45] ppgan.engine.trainer INFO: Iter: 16850/20000 lr: 9.993e-05 loss_promary: 10.765 loss_dual: 1.116 loss_total: 11.881 batch_cost: 0.24630 sec reader_cost: 0.00032 sec ips: 32.48052 images/s eta: 0:12:55
    [05/29 18:42:47] ppgan.engine.trainer INFO: Iter: 16860/20000 lr: 9.993e-05 loss_promary: 7.162 loss_dual: 0.968 loss_total: 8.130 batch_cost: 0.23493 sec reader_cost: 0.00032 sec ips: 34.05246 images/s eta: 0:12:17
    [05/29 18:42:50] ppgan.engine.trainer INFO: Iter: 16870/20000 lr: 9.993e-05 loss_promary: 7.413 loss_dual: 0.986 loss_total: 8.399 batch_cost: 0.22943 sec reader_cost: 0.00032 sec ips: 34.86934 images/s eta: 0:11:58
    [05/29 18:42:52] ppgan.engine.trainer INFO: Iter: 16880/20000 lr: 9.993e-05 loss_promary: 8.672 loss_dual: 1.072 loss_total: 9.743 batch_cost: 0.23229 sec reader_cost: 0.00033 sec ips: 34.43957 images/s eta: 0:12:04
    [05/29 18:42:54] ppgan.engine.trainer INFO: Iter: 16890/20000 lr: 9.993e-05 loss_promary: 10.472 loss_dual: 1.210 loss_total: 11.682 batch_cost: 0.23044 sec reader_cost: 0.00032 sec ips: 34.71669 images/s eta: 0:11:56
    [05/29 18:42:57] ppgan.engine.trainer INFO: Iter: 16900/20000 lr: 9.993e-05 loss_promary: 8.501 loss_dual: 0.949 loss_total: 9.450 batch_cost: 0.23334 sec reader_cost: 0.00033 sec ips: 34.28480 images/s eta: 0:12:03
    [05/29 18:42:59] ppgan.engine.trainer INFO: Iter: 16910/20000 lr: 9.993e-05 loss_promary: 8.310 loss_dual: 0.931 loss_total: 9.241 batch_cost: 0.23342 sec reader_cost: 0.00032 sec ips: 34.27262 images/s eta: 0:12:01
    [05/29 18:43:01] ppgan.engine.trainer INFO: Iter: 16920/20000 lr: 9.993e-05 loss_promary: 9.462 loss_dual: 1.099 loss_total: 10.561 batch_cost: 0.21573 sec reader_cost: 0.00031 sec ips: 37.08356 images/s eta: 0:11:04
    [05/29 18:43:03] ppgan.engine.trainer INFO: Iter: 16930/20000 lr: 9.993e-05 loss_promary: 11.399 loss_dual: 1.266 loss_total: 12.666 batch_cost: 0.20225 sec reader_cost: 0.00030 sec ips: 39.55510 images/s eta: 0:10:20
    [05/29 18:43:05] ppgan.engine.trainer INFO: Iter: 16940/20000 lr: 9.993e-05 loss_promary: 6.593 loss_dual: 0.809 loss_total: 7.402 batch_cost: 0.20271 sec reader_cost: 0.00030 sec ips: 39.46499 images/s eta: 0:10:20
    [05/29 18:43:07] ppgan.engine.trainer INFO: Iter: 16950/20000 lr: 9.993e-05 loss_promary: 6.208 loss_dual: 0.815 loss_total: 7.023 batch_cost: 0.20455 sec reader_cost: 0.00031 sec ips: 39.11030 images/s eta: 0:10:23
    [05/29 18:43:09] ppgan.engine.trainer INFO: Iter: 16960/20000 lr: 9.993e-05 loss_promary: 9.784 loss_dual: 1.071 loss_total: 10.855 batch_cost: 0.20206 sec reader_cost: 0.00029 sec ips: 39.59135 images/s eta: 0:10:14
    [05/29 18:43:11] ppgan.engine.trainer INFO: Iter: 16970/20000 lr: 9.993e-05 loss_promary: 9.620 loss_dual: 1.117 loss_total: 10.737 batch_cost: 0.20253 sec reader_cost: 0.00030 sec ips: 39.49993 images/s eta: 0:10:13
    [05/29 18:43:13] ppgan.engine.trainer INFO: Iter: 16980/20000 lr: 9.993e-05 loss_promary: 11.556 loss_dual: 1.322 loss_total: 12.878 batch_cost: 0.20070 sec reader_cost: 0.00031 sec ips: 39.86008 images/s eta: 0:10:06
    [05/29 18:43:15] ppgan.engine.trainer INFO: Iter: 16990/20000 lr: 9.993e-05 loss_promary: 11.158 loss_dual: 1.275 loss_total: 12.433 batch_cost: 0.20223 sec reader_cost: 0.00030 sec ips: 39.55891 images/s eta: 0:10:08
    [05/29 18:43:17] ppgan.engine.trainer INFO: Iter: 17000/20000 lr: 9.993e-05 loss_promary: 7.971 loss_dual: 0.996 loss_total: 8.967 batch_cost: 0.21884 sec reader_cost: 0.00030 sec ips: 36.55670 images/s eta: 0:10:56
    [05/29 18:43:17] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:43:19] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:43:21] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:43:23] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:43:24] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:43:26] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:43:28] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:43:30] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:43:31] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:43:33] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:43:35] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:43:37] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:43:38] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:43:40] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:43:42] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:43:44] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:43:45] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:43:47] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:43:49] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:43:51] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:43:52] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:43:54] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:43:56] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:43:58] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:43:59] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:44:01] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:44:03] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:44:04] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:44:06] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:44:08] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:44:10] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:44:11] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:44:13] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:44:15] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:44:17] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:44:19] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:44:21] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:44:22] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:44:24] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:44:26] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:44:28] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:44:29] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:44:31] ppgan.engine.trainer INFO: Metric psnr: 29.3015
    [05/29 18:44:31] ppgan.engine.trainer INFO: Metric ssim: 0.7969
    [05/29 18:44:33] ppgan.engine.trainer INFO: Iter: 17010/20000 lr: 9.993e-05 loss_promary: 7.897 loss_dual: 0.996 loss_total: 8.894 batch_cost: 0.21141 sec reader_cost: 0.00040 sec ips: 37.84031 images/s eta: 0:10:32
    [05/29 18:44:36] ppgan.engine.trainer INFO: Iter: 17020/20000 lr: 9.993e-05 loss_promary: 8.795 loss_dual: 0.970 loss_total: 9.765 batch_cost: 0.22467 sec reader_cost: 0.00032 sec ips: 35.60725 images/s eta: 0:11:09
    [05/29 18:44:38] ppgan.engine.trainer INFO: Iter: 17030/20000 lr: 9.993e-05 loss_promary: 5.299 loss_dual: 0.680 loss_total: 5.979 batch_cost: 0.20269 sec reader_cost: 0.00029 sec ips: 39.46855 images/s eta: 0:10:01
    [05/29 18:44:40] ppgan.engine.trainer INFO: Iter: 17040/20000 lr: 9.993e-05 loss_promary: 10.979 loss_dual: 1.161 loss_total: 12.140 batch_cost: 0.20194 sec reader_cost: 0.00029 sec ips: 39.61548 images/s eta: 0:09:57
    [05/29 18:44:42] ppgan.engine.trainer INFO: Iter: 17050/20000 lr: 9.993e-05 loss_promary: 9.458 loss_dual: 1.056 loss_total: 10.514 batch_cost: 0.20157 sec reader_cost: 0.00030 sec ips: 39.68869 images/s eta: 0:09:54
    [05/29 18:44:44] ppgan.engine.trainer INFO: Iter: 17060/20000 lr: 9.993e-05 loss_promary: 8.775 loss_dual: 0.974 loss_total: 9.749 batch_cost: 0.20086 sec reader_cost: 0.00030 sec ips: 39.82878 images/s eta: 0:09:50
    [05/29 18:44:46] ppgan.engine.trainer INFO: Iter: 17070/20000 lr: 9.993e-05 loss_promary: 10.204 loss_dual: 1.106 loss_total: 11.310 batch_cost: 0.20292 sec reader_cost: 0.00031 sec ips: 39.42423 images/s eta: 0:09:54
    [05/29 18:44:48] ppgan.engine.trainer INFO: Iter: 17080/20000 lr: 9.993e-05 loss_promary: 10.603 loss_dual: 1.101 loss_total: 11.704 batch_cost: 0.20624 sec reader_cost: 0.00030 sec ips: 38.78979 images/s eta: 0:10:02
    [05/29 18:44:50] ppgan.engine.trainer INFO: Iter: 17090/20000 lr: 9.993e-05 loss_promary: 8.984 loss_dual: 1.013 loss_total: 9.997 batch_cost: 0.20947 sec reader_cost: 0.00030 sec ips: 38.19202 images/s eta: 0:10:09
    [05/29 18:44:52] ppgan.engine.trainer INFO: Iter: 17100/20000 lr: 9.993e-05 loss_promary: 8.399 loss_dual: 1.056 loss_total: 9.454 batch_cost: 0.21783 sec reader_cost: 0.00031 sec ips: 36.72551 images/s eta: 0:10:31
    [05/29 18:44:54] ppgan.engine.trainer INFO: Iter: 17110/20000 lr: 9.993e-05 loss_promary: 9.620 loss_dual: 1.138 loss_total: 10.757 batch_cost: 0.21359 sec reader_cost: 0.00032 sec ips: 37.45514 images/s eta: 0:10:17
    [05/29 18:44:56] ppgan.engine.trainer INFO: Iter: 17120/20000 lr: 9.993e-05 loss_promary: 9.217 loss_dual: 1.019 loss_total: 10.236 batch_cost: 0.20585 sec reader_cost: 0.00030 sec ips: 38.86285 images/s eta: 0:09:52
    [05/29 18:44:58] ppgan.engine.trainer INFO: Iter: 17130/20000 lr: 9.993e-05 loss_promary: 8.395 loss_dual: 0.997 loss_total: 9.392 batch_cost: 0.20457 sec reader_cost: 0.00030 sec ips: 39.10667 images/s eta: 0:09:47
    [05/29 18:45:00] ppgan.engine.trainer INFO: Iter: 17140/20000 lr: 9.993e-05 loss_promary: 10.170 loss_dual: 1.066 loss_total: 11.236 batch_cost: 0.20375 sec reader_cost: 0.00030 sec ips: 39.26333 images/s eta: 0:09:42
    [05/29 18:45:02] ppgan.engine.trainer INFO: Iter: 17150/20000 lr: 9.993e-05 loss_promary: 9.443 loss_dual: 1.040 loss_total: 10.483 batch_cost: 0.20307 sec reader_cost: 0.00030 sec ips: 39.39619 images/s eta: 0:09:38
    [05/29 18:45:04] ppgan.engine.trainer INFO: Iter: 17160/20000 lr: 9.993e-05 loss_promary: 11.126 loss_dual: 1.188 loss_total: 12.315 batch_cost: 0.20192 sec reader_cost: 0.00030 sec ips: 39.61922 images/s eta: 0:09:33
    [05/29 18:45:06] ppgan.engine.trainer INFO: Iter: 17170/20000 lr: 9.993e-05 loss_promary: 10.384 loss_dual: 1.129 loss_total: 11.514 batch_cost: 0.20377 sec reader_cost: 0.00029 sec ips: 39.26074 images/s eta: 0:09:36
    [05/29 18:45:08] ppgan.engine.trainer INFO: Iter: 17180/20000 lr: 9.993e-05 loss_promary: 8.270 loss_dual: 1.063 loss_total: 9.332 batch_cost: 0.20059 sec reader_cost: 0.00029 sec ips: 39.88160 images/s eta: 0:09:25
    [05/29 18:45:11] ppgan.engine.trainer INFO: Iter: 17190/20000 lr: 9.993e-05 loss_promary: 8.170 loss_dual: 0.948 loss_total: 9.118 batch_cost: 0.20411 sec reader_cost: 0.00029 sec ips: 39.19425 images/s eta: 0:09:33
    [05/29 18:45:13] ppgan.engine.trainer INFO: Iter: 17200/20000 lr: 9.993e-05 loss_promary: 8.704 loss_dual: 0.968 loss_total: 9.672 batch_cost: 0.20317 sec reader_cost: 0.00029 sec ips: 39.37550 images/s eta: 0:09:28
    [05/29 18:45:15] ppgan.engine.trainer INFO: Iter: 17210/20000 lr: 9.993e-05 loss_promary: 8.303 loss_dual: 0.905 loss_total: 9.208 batch_cost: 0.20292 sec reader_cost: 0.00029 sec ips: 39.42408 images/s eta: 0:09:26
    [05/29 18:45:17] ppgan.engine.trainer INFO: Iter: 17220/20000 lr: 9.993e-05 loss_promary: 10.869 loss_dual: 1.192 loss_total: 12.061 batch_cost: 0.20551 sec reader_cost: 0.00030 sec ips: 38.92777 images/s eta: 0:09:31
    [05/29 18:45:19] ppgan.engine.trainer INFO: Iter: 17230/20000 lr: 9.993e-05 loss_promary: 10.074 loss_dual: 1.139 loss_total: 11.214 batch_cost: 0.20321 sec reader_cost: 0.00029 sec ips: 39.36750 images/s eta: 0:09:22
    [05/29 18:45:21] ppgan.engine.trainer INFO: Iter: 17240/20000 lr: 9.993e-05 loss_promary: 9.463 loss_dual: 1.022 loss_total: 10.485 batch_cost: 0.20220 sec reader_cost: 0.00030 sec ips: 39.56396 images/s eta: 0:09:18
    [05/29 18:45:23] ppgan.engine.trainer INFO: Iter: 17250/20000 lr: 9.993e-05 loss_promary: 8.803 loss_dual: 1.038 loss_total: 9.841 batch_cost: 0.20192 sec reader_cost: 0.00030 sec ips: 39.61898 images/s eta: 0:09:15
    [05/29 18:45:25] ppgan.engine.trainer INFO: Iter: 17260/20000 lr: 9.993e-05 loss_promary: 9.244 loss_dual: 1.043 loss_total: 10.287 batch_cost: 0.21691 sec reader_cost: 0.00031 sec ips: 36.88181 images/s eta: 0:09:54
    [05/29 18:45:27] ppgan.engine.trainer INFO: Iter: 17270/20000 lr: 9.993e-05 loss_promary: 10.264 loss_dual: 1.153 loss_total: 11.418 batch_cost: 0.20164 sec reader_cost: 0.00030 sec ips: 39.67392 images/s eta: 0:09:10
    [05/29 18:45:29] ppgan.engine.trainer INFO: Iter: 17280/20000 lr: 9.993e-05 loss_promary: 8.568 loss_dual: 0.977 loss_total: 9.545 batch_cost: 0.20165 sec reader_cost: 0.00030 sec ips: 39.67184 images/s eta: 0:09:08
    [05/29 18:45:31] ppgan.engine.trainer INFO: Iter: 17290/20000 lr: 9.993e-05 loss_promary: 8.011 loss_dual: 0.951 loss_total: 8.961 batch_cost: 0.20427 sec reader_cost: 0.00031 sec ips: 39.16290 images/s eta: 0:09:13
    [05/29 18:45:33] ppgan.engine.trainer INFO: Iter: 17300/20000 lr: 9.993e-05 loss_promary: 9.028 loss_dual: 0.999 loss_total: 10.027 batch_cost: 0.20192 sec reader_cost: 0.00030 sec ips: 39.62011 images/s eta: 0:09:05
    [05/29 18:45:35] ppgan.engine.trainer INFO: Iter: 17310/20000 lr: 9.993e-05 loss_promary: 7.754 loss_dual: 0.960 loss_total: 8.714 batch_cost: 0.21235 sec reader_cost: 0.00031 sec ips: 37.67337 images/s eta: 0:09:31
    [05/29 18:45:37] ppgan.engine.trainer INFO: Iter: 17320/20000 lr: 9.993e-05 loss_promary: 7.483 loss_dual: 0.898 loss_total: 8.382 batch_cost: 0.23509 sec reader_cost: 0.00034 sec ips: 34.02914 images/s eta: 0:10:30
    [05/29 18:45:40] ppgan.engine.trainer INFO: Iter: 17330/20000 lr: 9.993e-05 loss_promary: 11.700 loss_dual: 1.336 loss_total: 13.036 batch_cost: 0.24014 sec reader_cost: 0.00036 sec ips: 33.31407 images/s eta: 0:10:41
    [05/29 18:45:42] ppgan.engine.trainer INFO: Iter: 17340/20000 lr: 9.993e-05 loss_promary: 11.918 loss_dual: 1.337 loss_total: 13.255 batch_cost: 0.21767 sec reader_cost: 0.00031 sec ips: 36.75276 images/s eta: 0:09:39
    [05/29 18:45:44] ppgan.engine.trainer INFO: Iter: 17350/20000 lr: 9.993e-05 loss_promary: 8.456 loss_dual: 1.041 loss_total: 9.497 batch_cost: 0.20610 sec reader_cost: 0.00031 sec ips: 38.81533 images/s eta: 0:09:06
    [05/29 18:45:46] ppgan.engine.trainer INFO: Iter: 17360/20000 lr: 9.993e-05 loss_promary: 9.253 loss_dual: 1.146 loss_total: 10.399 batch_cost: 0.20354 sec reader_cost: 0.00029 sec ips: 39.30490 images/s eta: 0:08:57
    [05/29 18:45:48] ppgan.engine.trainer INFO: Iter: 17370/20000 lr: 9.993e-05 loss_promary: 6.461 loss_dual: 0.794 loss_total: 7.255 batch_cost: 0.20219 sec reader_cost: 0.00029 sec ips: 39.56625 images/s eta: 0:08:51
    [05/29 18:45:50] ppgan.engine.trainer INFO: Iter: 17380/20000 lr: 9.993e-05 loss_promary: 7.698 loss_dual: 0.879 loss_total: 8.577 batch_cost: 0.21199 sec reader_cost: 0.00030 sec ips: 37.73765 images/s eta: 0:09:15
    [05/29 18:45:52] ppgan.engine.trainer INFO: Iter: 17390/20000 lr: 9.993e-05 loss_promary: 6.821 loss_dual: 0.822 loss_total: 7.643 batch_cost: 0.20612 sec reader_cost: 0.00031 sec ips: 38.81254 images/s eta: 0:08:57
    [05/29 18:45:54] ppgan.engine.trainer INFO: Iter: 17400/20000 lr: 9.993e-05 loss_promary: 9.240 loss_dual: 1.056 loss_total: 10.296 batch_cost: 0.20566 sec reader_cost: 0.00031 sec ips: 38.89963 images/s eta: 0:08:54
    [05/29 18:45:57] ppgan.engine.trainer INFO: Iter: 17410/20000 lr: 9.993e-05 loss_promary: 7.615 loss_dual: 0.949 loss_total: 8.564 batch_cost: 0.22771 sec reader_cost: 0.00031 sec ips: 35.13271 images/s eta: 0:09:49
    [05/29 18:45:59] ppgan.engine.trainer INFO: Iter: 17420/20000 lr: 9.993e-05 loss_promary: 9.322 loss_dual: 1.064 loss_total: 10.386 batch_cost: 0.20729 sec reader_cost: 0.00030 sec ips: 38.59413 images/s eta: 0:08:54
    [05/29 18:46:01] ppgan.engine.trainer INFO: Iter: 17430/20000 lr: 9.993e-05 loss_promary: 11.851 loss_dual: 1.265 loss_total: 13.116 batch_cost: 0.20765 sec reader_cost: 0.00030 sec ips: 38.52649 images/s eta: 0:08:53
    [05/29 18:46:03] ppgan.engine.trainer INFO: Iter: 17440/20000 lr: 9.993e-05 loss_promary: 7.749 loss_dual: 0.875 loss_total: 8.624 batch_cost: 0.20402 sec reader_cost: 0.00031 sec ips: 39.21261 images/s eta: 0:08:42
    [05/29 18:46:05] ppgan.engine.trainer INFO: Iter: 17450/20000 lr: 9.992e-05 loss_promary: 10.059 loss_dual: 1.049 loss_total: 11.108 batch_cost: 0.20564 sec reader_cost: 0.00030 sec ips: 38.90253 images/s eta: 0:08:44
    [05/29 18:46:07] ppgan.engine.trainer INFO: Iter: 17460/20000 lr: 9.992e-05 loss_promary: 7.848 loss_dual: 0.946 loss_total: 8.793 batch_cost: 0.20484 sec reader_cost: 0.00030 sec ips: 39.05520 images/s eta: 0:08:40
    [05/29 18:46:09] ppgan.engine.trainer INFO: Iter: 17470/20000 lr: 9.992e-05 loss_promary: 10.962 loss_dual: 1.194 loss_total: 12.155 batch_cost: 0.20415 sec reader_cost: 0.00030 sec ips: 39.18649 images/s eta: 0:08:36
    [05/29 18:46:11] ppgan.engine.trainer INFO: Iter: 17480/20000 lr: 9.992e-05 loss_promary: 9.371 loss_dual: 1.044 loss_total: 10.415 batch_cost: 0.20603 sec reader_cost: 0.00030 sec ips: 38.82953 images/s eta: 0:08:39
    [05/29 18:46:13] ppgan.engine.trainer INFO: Iter: 17490/20000 lr: 9.992e-05 loss_promary: 7.661 loss_dual: 0.919 loss_total: 8.579 batch_cost: 0.20360 sec reader_cost: 0.00030 sec ips: 39.29270 images/s eta: 0:08:31
    [05/29 18:46:15] ppgan.engine.trainer INFO: Iter: 17500/20000 lr: 9.992e-05 loss_promary: 7.667 loss_dual: 0.938 loss_total: 8.605 batch_cost: 0.20403 sec reader_cost: 0.00030 sec ips: 39.21057 images/s eta: 0:08:30
    [05/29 18:46:17] ppgan.engine.trainer INFO: Iter: 17510/20000 lr: 9.992e-05 loss_promary: 9.613 loss_dual: 1.063 loss_total: 10.676 batch_cost: 0.20242 sec reader_cost: 0.00031 sec ips: 39.52175 images/s eta: 0:08:24
    [05/29 18:46:19] ppgan.engine.trainer INFO: Iter: 17520/20000 lr: 9.992e-05 loss_promary: 7.811 loss_dual: 0.856 loss_total: 8.667 batch_cost: 0.20450 sec reader_cost: 0.00030 sec ips: 39.11935 images/s eta: 0:08:27
    [05/29 18:46:21] ppgan.engine.trainer INFO: Iter: 17530/20000 lr: 9.992e-05 loss_promary: 9.166 loss_dual: 1.036 loss_total: 10.202 batch_cost: 0.20614 sec reader_cost: 0.00030 sec ips: 38.80801 images/s eta: 0:08:29
    [05/29 18:46:23] ppgan.engine.trainer INFO: Iter: 17540/20000 lr: 9.992e-05 loss_promary: 9.182 loss_dual: 1.095 loss_total: 10.276 batch_cost: 0.20491 sec reader_cost: 0.00029 sec ips: 39.04118 images/s eta: 0:08:24
    [05/29 18:46:26] ppgan.engine.trainer INFO: Iter: 17550/20000 lr: 9.992e-05 loss_promary: 9.696 loss_dual: 1.113 loss_total: 10.810 batch_cost: 0.20655 sec reader_cost: 0.00030 sec ips: 38.73143 images/s eta: 0:08:26
    [05/29 18:46:28] ppgan.engine.trainer INFO: Iter: 17560/20000 lr: 9.992e-05 loss_promary: 8.132 loss_dual: 1.014 loss_total: 9.146 batch_cost: 0.21933 sec reader_cost: 0.00031 sec ips: 36.47527 images/s eta: 0:08:55
    [05/29 18:46:30] ppgan.engine.trainer INFO: Iter: 17570/20000 lr: 9.992e-05 loss_promary: 8.012 loss_dual: 0.940 loss_total: 8.952 batch_cost: 0.21356 sec reader_cost: 0.00031 sec ips: 37.46025 images/s eta: 0:08:38
    [05/29 18:46:32] ppgan.engine.trainer INFO: Iter: 17580/20000 lr: 9.992e-05 loss_promary: 8.285 loss_dual: 0.961 loss_total: 9.246 batch_cost: 0.20349 sec reader_cost: 0.00030 sec ips: 39.31354 images/s eta: 0:08:12
    [05/29 18:46:34] ppgan.engine.trainer INFO: Iter: 17590/20000 lr: 9.992e-05 loss_promary: 7.902 loss_dual: 0.890 loss_total: 8.792 batch_cost: 0.20646 sec reader_cost: 0.00031 sec ips: 38.74798 images/s eta: 0:08:17
    [05/29 18:46:36] ppgan.engine.trainer INFO: Iter: 17600/20000 lr: 9.992e-05 loss_promary: 10.267 loss_dual: 1.089 loss_total: 11.356 batch_cost: 0.21022 sec reader_cost: 0.00030 sec ips: 38.05530 images/s eta: 0:08:24
    [05/29 18:46:38] ppgan.engine.trainer INFO: Iter: 17610/20000 lr: 9.992e-05 loss_promary: 8.879 loss_dual: 1.028 loss_total: 9.907 batch_cost: 0.20774 sec reader_cost: 0.00031 sec ips: 38.50888 images/s eta: 0:08:16
    [05/29 18:46:40] ppgan.engine.trainer INFO: Iter: 17620/20000 lr: 9.992e-05 loss_promary: 9.985 loss_dual: 1.112 loss_total: 11.097 batch_cost: 0.20325 sec reader_cost: 0.00030 sec ips: 39.36062 images/s eta: 0:08:03
    [05/29 18:46:42] ppgan.engine.trainer INFO: Iter: 17630/20000 lr: 9.992e-05 loss_promary: 9.984 loss_dual: 1.246 loss_total: 11.230 batch_cost: 0.20513 sec reader_cost: 0.00029 sec ips: 38.99933 images/s eta: 0:08:06
    [05/29 18:46:44] ppgan.engine.trainer INFO: Iter: 17640/20000 lr: 9.992e-05 loss_promary: 7.784 loss_dual: 0.929 loss_total: 8.713 batch_cost: 0.20569 sec reader_cost: 0.00037 sec ips: 38.89360 images/s eta: 0:08:05
    [05/29 18:46:47] ppgan.engine.trainer INFO: Iter: 17650/20000 lr: 9.992e-05 loss_promary: 8.377 loss_dual: 1.020 loss_total: 9.396 batch_cost: 0.25881 sec reader_cost: 0.04542 sec ips: 30.91117 images/s eta: 0:10:08
    [05/29 18:46:49] ppgan.engine.trainer INFO: Iter: 17660/20000 lr: 9.992e-05 loss_promary: 7.042 loss_dual: 0.913 loss_total: 7.956 batch_cost: 0.19918 sec reader_cost: 0.00029 sec ips: 40.16541 images/s eta: 0:07:46
    [05/29 18:46:51] ppgan.engine.trainer INFO: Iter: 17670/20000 lr: 9.992e-05 loss_promary: 7.635 loss_dual: 0.983 loss_total: 8.617 batch_cost: 0.20245 sec reader_cost: 0.00029 sec ips: 39.51533 images/s eta: 0:07:51
    [05/29 18:46:53] ppgan.engine.trainer INFO: Iter: 17680/20000 lr: 9.992e-05 loss_promary: 10.961 loss_dual: 1.176 loss_total: 12.137 batch_cost: 0.20191 sec reader_cost: 0.00031 sec ips: 39.62192 images/s eta: 0:07:48
    [05/29 18:46:55] ppgan.engine.trainer INFO: Iter: 17690/20000 lr: 9.992e-05 loss_promary: 9.282 loss_dual: 1.045 loss_total: 10.327 batch_cost: 0.20363 sec reader_cost: 0.00032 sec ips: 39.28641 images/s eta: 0:07:50
    [05/29 18:46:57] ppgan.engine.trainer INFO: Iter: 17700/20000 lr: 9.992e-05 loss_promary: 7.776 loss_dual: 0.923 loss_total: 8.699 batch_cost: 0.20633 sec reader_cost: 0.00031 sec ips: 38.77290 images/s eta: 0:07:54
    [05/29 18:46:59] ppgan.engine.trainer INFO: Iter: 17710/20000 lr: 9.992e-05 loss_promary: 8.429 loss_dual: 1.011 loss_total: 9.440 batch_cost: 0.21396 sec reader_cost: 0.00030 sec ips: 37.39067 images/s eta: 0:08:09
    [05/29 18:47:01] ppgan.engine.trainer INFO: Iter: 17720/20000 lr: 9.992e-05 loss_promary: 8.639 loss_dual: 1.039 loss_total: 9.678 batch_cost: 0.21479 sec reader_cost: 0.00029 sec ips: 37.24536 images/s eta: 0:08:09
    [05/29 18:47:03] ppgan.engine.trainer INFO: Iter: 17730/20000 lr: 9.992e-05 loss_promary: 10.766 loss_dual: 1.237 loss_total: 12.004 batch_cost: 0.20078 sec reader_cost: 0.00029 sec ips: 39.84451 images/s eta: 0:07:35
    [05/29 18:47:05] ppgan.engine.trainer INFO: Iter: 17740/20000 lr: 9.992e-05 loss_promary: 11.039 loss_dual: 1.141 loss_total: 12.180 batch_cost: 0.20254 sec reader_cost: 0.00028 sec ips: 39.49825 images/s eta: 0:07:37
    [05/29 18:47:07] ppgan.engine.trainer INFO: Iter: 17750/20000 lr: 9.992e-05 loss_promary: 7.681 loss_dual: 0.877 loss_total: 8.558 batch_cost: 0.20407 sec reader_cost: 0.00029 sec ips: 39.20177 images/s eta: 0:07:39
    [05/29 18:47:09] ppgan.engine.trainer INFO: Iter: 17760/20000 lr: 9.992e-05 loss_promary: 9.097 loss_dual: 1.008 loss_total: 10.105 batch_cost: 0.19989 sec reader_cost: 0.00029 sec ips: 40.02251 images/s eta: 0:07:27
    [05/29 18:47:11] ppgan.engine.trainer INFO: Iter: 17770/20000 lr: 9.992e-05 loss_promary: 11.352 loss_dual: 1.232 loss_total: 12.584 batch_cost: 0.20306 sec reader_cost: 0.00029 sec ips: 39.39686 images/s eta: 0:07:32
    [05/29 18:47:13] ppgan.engine.trainer INFO: Iter: 17780/20000 lr: 9.992e-05 loss_promary: 9.113 loss_dual: 1.125 loss_total: 10.237 batch_cost: 0.20120 sec reader_cost: 0.00029 sec ips: 39.76081 images/s eta: 0:07:26
    [05/29 18:47:15] ppgan.engine.trainer INFO: Iter: 17790/20000 lr: 9.992e-05 loss_promary: 12.315 loss_dual: 1.326 loss_total: 13.641 batch_cost: 0.20030 sec reader_cost: 0.00029 sec ips: 39.93982 images/s eta: 0:07:22
    [05/29 18:47:17] ppgan.engine.trainer INFO: Iter: 17800/20000 lr: 9.992e-05 loss_promary: 7.918 loss_dual: 1.016 loss_total: 8.934 batch_cost: 0.20031 sec reader_cost: 0.00028 sec ips: 39.93768 images/s eta: 0:07:20
    [05/29 18:47:19] ppgan.engine.trainer INFO: Iter: 17810/20000 lr: 9.992e-05 loss_promary: 8.822 loss_dual: 0.995 loss_total: 9.818 batch_cost: 0.20022 sec reader_cost: 0.00028 sec ips: 39.95685 images/s eta: 0:07:18
    [05/29 18:47:22] ppgan.engine.trainer INFO: Iter: 17820/20000 lr: 9.992e-05 loss_promary: 7.662 loss_dual: 0.919 loss_total: 8.581 batch_cost: 0.20304 sec reader_cost: 0.00029 sec ips: 39.40157 images/s eta: 0:07:22
    [05/29 18:47:24] ppgan.engine.trainer INFO: Iter: 17830/20000 lr: 9.992e-05 loss_promary: 7.996 loss_dual: 0.970 loss_total: 8.966 batch_cost: 0.20093 sec reader_cost: 0.00029 sec ips: 39.81505 images/s eta: 0:07:16
    [05/29 18:47:26] ppgan.engine.trainer INFO: Iter: 17840/20000 lr: 9.992e-05 loss_promary: 8.954 loss_dual: 1.088 loss_total: 10.042 batch_cost: 0.20203 sec reader_cost: 0.00031 sec ips: 39.59842 images/s eta: 0:07:16
    [05/29 18:47:28] ppgan.engine.trainer INFO: Iter: 17850/20000 lr: 9.992e-05 loss_promary: 8.666 loss_dual: 1.133 loss_total: 9.799 batch_cost: 0.20248 sec reader_cost: 0.00030 sec ips: 39.51012 images/s eta: 0:07:15
    [05/29 18:47:30] ppgan.engine.trainer INFO: Iter: 17860/20000 lr: 9.992e-05 loss_promary: 9.993 loss_dual: 1.110 loss_total: 11.103 batch_cost: 0.20376 sec reader_cost: 0.00031 sec ips: 39.26279 images/s eta: 0:07:16
    [05/29 18:47:32] ppgan.engine.trainer INFO: Iter: 17870/20000 lr: 9.992e-05 loss_promary: 9.583 loss_dual: 1.103 loss_total: 10.686 batch_cost: 0.21364 sec reader_cost: 0.00029 sec ips: 37.44591 images/s eta: 0:07:35
    [05/29 18:47:34] ppgan.engine.trainer INFO: Iter: 17880/20000 lr: 9.992e-05 loss_promary: 12.866 loss_dual: 1.322 loss_total: 14.188 batch_cost: 0.20539 sec reader_cost: 0.00030 sec ips: 38.95111 images/s eta: 0:07:15
    [05/29 18:47:36] ppgan.engine.trainer INFO: Iter: 17890/20000 lr: 9.992e-05 loss_promary: 12.870 loss_dual: 1.410 loss_total: 14.280 batch_cost: 0.20140 sec reader_cost: 0.00030 sec ips: 39.72156 images/s eta: 0:07:04
    [05/29 18:47:38] ppgan.engine.trainer INFO: Iter: 17900/20000 lr: 9.992e-05 loss_promary: 7.373 loss_dual: 0.914 loss_total: 8.287 batch_cost: 0.20870 sec reader_cost: 0.00030 sec ips: 38.33314 images/s eta: 0:07:18
    [05/29 18:47:40] ppgan.engine.trainer INFO: Iter: 17910/20000 lr: 9.992e-05 loss_promary: 8.026 loss_dual: 0.950 loss_total: 8.976 batch_cost: 0.20241 sec reader_cost: 0.00030 sec ips: 39.52341 images/s eta: 0:07:03
    [05/29 18:47:42] ppgan.engine.trainer INFO: Iter: 17920/20000 lr: 9.992e-05 loss_promary: 12.745 loss_dual: 1.280 loss_total: 14.026 batch_cost: 0.20239 sec reader_cost: 0.00030 sec ips: 39.52826 images/s eta: 0:07:00
    [05/29 18:47:44] ppgan.engine.trainer INFO: Iter: 17930/20000 lr: 9.992e-05 loss_promary: 11.563 loss_dual: 1.249 loss_total: 12.812 batch_cost: 0.20007 sec reader_cost: 0.00028 sec ips: 39.98645 images/s eta: 0:06:54
    [05/29 18:47:46] ppgan.engine.trainer INFO: Iter: 17940/20000 lr: 9.992e-05 loss_promary: 6.573 loss_dual: 0.853 loss_total: 7.426 batch_cost: 0.20099 sec reader_cost: 0.00030 sec ips: 39.80288 images/s eta: 0:06:54
    [05/29 18:47:48] ppgan.engine.trainer INFO: Iter: 17950/20000 lr: 9.992e-05 loss_promary: 10.346 loss_dual: 1.033 loss_total: 11.379 batch_cost: 0.20329 sec reader_cost: 0.00031 sec ips: 39.35345 images/s eta: 0:06:56
    [05/29 18:47:50] ppgan.engine.trainer INFO: Iter: 17960/20000 lr: 9.992e-05 loss_promary: 8.568 loss_dual: 0.957 loss_total: 9.525 batch_cost: 0.23775 sec reader_cost: 0.00034 sec ips: 33.64859 images/s eta: 0:08:05
    [05/29 18:47:53] ppgan.engine.trainer INFO: Iter: 17970/20000 lr: 9.992e-05 loss_promary: 12.696 loss_dual: 1.345 loss_total: 14.041 batch_cost: 0.23346 sec reader_cost: 0.00033 sec ips: 34.26641 images/s eta: 0:07:53
    [05/29 18:47:55] ppgan.engine.trainer INFO: Iter: 17980/20000 lr: 9.992e-05 loss_promary: 8.311 loss_dual: 0.941 loss_total: 9.253 batch_cost: 0.23391 sec reader_cost: 0.00033 sec ips: 34.20120 images/s eta: 0:07:52
    [05/29 18:47:57] ppgan.engine.trainer INFO: Iter: 17990/20000 lr: 9.992e-05 loss_promary: 6.975 loss_dual: 0.905 loss_total: 7.880 batch_cost: 0.23578 sec reader_cost: 0.00034 sec ips: 33.92927 images/s eta: 0:07:53
    [05/29 18:48:00] ppgan.engine.trainer INFO: Iter: 18000/20000 lr: 9.992e-05 loss_promary: 8.036 loss_dual: 0.975 loss_total: 9.010 batch_cost: 0.23508 sec reader_cost: 0.00032 sec ips: 34.03137 images/s eta: 0:07:50
    [05/29 18:48:00] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:48:02] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:48:04] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:48:05] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:48:07] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:48:09] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:48:11] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:48:12] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:48:14] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:48:16] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:48:17] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:48:19] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:48:21] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:48:23] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:48:24] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:48:26] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:48:28] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:48:30] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:48:31] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:48:33] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:48:35] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:48:37] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:48:38] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:48:40] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:48:42] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:48:43] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:48:45] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:48:47] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:48:49] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:48:50] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:48:52] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:48:54] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:48:56] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:48:57] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:48:59] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:49:01] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:49:02] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:49:04] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:49:06] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:49:08] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:49:09] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:49:11] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:49:13] ppgan.engine.trainer INFO: Metric psnr: 29.3801
    [05/29 18:49:13] ppgan.engine.trainer INFO: Metric ssim: 0.8049
    [05/29 18:49:15] ppgan.engine.trainer INFO: Iter: 18010/20000 lr: 9.992e-05 loss_promary: 11.052 loss_dual: 1.135 loss_total: 12.187 batch_cost: 0.20718 sec reader_cost: 0.00036 sec ips: 38.61310 images/s eta: 0:06:52
    [05/29 18:49:17] ppgan.engine.trainer INFO: Iter: 18020/20000 lr: 9.992e-05 loss_promary: 10.168 loss_dual: 1.085 loss_total: 11.253 batch_cost: 0.20125 sec reader_cost: 0.00030 sec ips: 39.75083 images/s eta: 0:06:38
    [05/29 18:49:19] ppgan.engine.trainer INFO: Iter: 18030/20000 lr: 9.992e-05 loss_promary: 9.813 loss_dual: 1.139 loss_total: 10.952 batch_cost: 0.20056 sec reader_cost: 0.00030 sec ips: 39.88773 images/s eta: 0:06:35
    [05/29 18:49:21] ppgan.engine.trainer INFO: Iter: 18040/20000 lr: 9.992e-05 loss_promary: 9.116 loss_dual: 1.099 loss_total: 10.214 batch_cost: 0.20464 sec reader_cost: 0.00031 sec ips: 39.09319 images/s eta: 0:06:41
    [05/29 18:49:23] ppgan.engine.trainer INFO: Iter: 18050/20000 lr: 9.992e-05 loss_promary: 9.988 loss_dual: 1.157 loss_total: 11.145 batch_cost: 0.20307 sec reader_cost: 0.00031 sec ips: 39.39594 images/s eta: 0:06:35
    [05/29 18:49:25] ppgan.engine.trainer INFO: Iter: 18060/20000 lr: 9.992e-05 loss_promary: 8.114 loss_dual: 0.984 loss_total: 9.098 batch_cost: 0.20226 sec reader_cost: 0.00030 sec ips: 39.55212 images/s eta: 0:06:32
    [05/29 18:49:27] ppgan.engine.trainer INFO: Iter: 18070/20000 lr: 9.992e-05 loss_promary: 7.133 loss_dual: 0.986 loss_total: 8.119 batch_cost: 0.20232 sec reader_cost: 0.00029 sec ips: 39.54214 images/s eta: 0:06:30
    [05/29 18:49:29] ppgan.engine.trainer INFO: Iter: 18080/20000 lr: 9.992e-05 loss_promary: 8.977 loss_dual: 1.128 loss_total: 10.105 batch_cost: 0.20116 sec reader_cost: 0.00030 sec ips: 39.76843 images/s eta: 0:06:26
    [05/29 18:49:31] ppgan.engine.trainer INFO: Iter: 18090/20000 lr: 9.992e-05 loss_promary: 7.236 loss_dual: 0.906 loss_total: 8.142 batch_cost: 0.20419 sec reader_cost: 0.00030 sec ips: 39.17837 images/s eta: 0:06:30
    [05/29 18:49:33] ppgan.engine.trainer INFO: Iter: 18100/20000 lr: 9.992e-05 loss_promary: 11.996 loss_dual: 1.269 loss_total: 13.265 batch_cost: 0.20341 sec reader_cost: 0.00030 sec ips: 39.33029 images/s eta: 0:06:26
    [05/29 18:49:35] ppgan.engine.trainer INFO: Iter: 18110/20000 lr: 9.992e-05 loss_promary: 10.069 loss_dual: 1.052 loss_total: 11.121 batch_cost: 0.20319 sec reader_cost: 0.00030 sec ips: 39.37263 images/s eta: 0:06:24
    [05/29 18:49:37] ppgan.engine.trainer INFO: Iter: 18120/20000 lr: 9.992e-05 loss_promary: 11.818 loss_dual: 1.294 loss_total: 13.112 batch_cost: 0.20560 sec reader_cost: 0.00029 sec ips: 38.91138 images/s eta: 0:06:26
    [05/29 18:49:40] ppgan.engine.trainer INFO: Iter: 18130/20000 lr: 9.992e-05 loss_promary: 10.240 loss_dual: 1.177 loss_total: 11.416 batch_cost: 0.21895 sec reader_cost: 0.00029 sec ips: 36.53844 images/s eta: 0:06:49
    [05/29 18:49:42] ppgan.engine.trainer INFO: Iter: 18140/20000 lr: 9.992e-05 loss_promary: 10.241 loss_dual: 1.151 loss_total: 11.392 batch_cost: 0.20602 sec reader_cost: 0.00030 sec ips: 38.83152 images/s eta: 0:06:23
    [05/29 18:49:44] ppgan.engine.trainer INFO: Iter: 18150/20000 lr: 9.992e-05 loss_promary: 11.056 loss_dual: 1.142 loss_total: 12.198 batch_cost: 0.20208 sec reader_cost: 0.00031 sec ips: 39.58770 images/s eta: 0:06:13
    [05/29 18:49:46] ppgan.engine.trainer INFO: Iter: 18160/20000 lr: 9.992e-05 loss_promary: 9.239 loss_dual: 1.091 loss_total: 10.329 batch_cost: 0.20312 sec reader_cost: 0.00030 sec ips: 39.38597 images/s eta: 0:06:13
    [05/29 18:49:48] ppgan.engine.trainer INFO: Iter: 18170/20000 lr: 9.992e-05 loss_promary: 9.092 loss_dual: 1.055 loss_total: 10.147 batch_cost: 0.20242 sec reader_cost: 0.00029 sec ips: 39.52147 images/s eta: 0:06:10
    [05/29 18:49:50] ppgan.engine.trainer INFO: Iter: 18180/20000 lr: 9.992e-05 loss_promary: 8.626 loss_dual: 0.969 loss_total: 9.595 batch_cost: 0.20698 sec reader_cost: 0.00030 sec ips: 38.65094 images/s eta: 0:06:16
    [05/29 18:49:52] ppgan.engine.trainer INFO: Iter: 18190/20000 lr: 9.992e-05 loss_promary: 11.144 loss_dual: 1.173 loss_total: 12.318 batch_cost: 0.20609 sec reader_cost: 0.00031 sec ips: 38.81744 images/s eta: 0:06:13
    [05/29 18:49:54] ppgan.engine.trainer INFO: Iter: 18200/20000 lr: 9.992e-05 loss_promary: 9.396 loss_dual: 1.035 loss_total: 10.431 batch_cost: 0.20624 sec reader_cost: 0.00032 sec ips: 38.78971 images/s eta: 0:06:11
    [05/29 18:49:56] ppgan.engine.trainer INFO: Iter: 18210/20000 lr: 9.992e-05 loss_promary: 9.343 loss_dual: 1.063 loss_total: 10.406 batch_cost: 0.20519 sec reader_cost: 0.00031 sec ips: 38.98813 images/s eta: 0:06:07
    [05/29 18:49:58] ppgan.engine.trainer INFO: Iter: 18220/20000 lr: 9.992e-05 loss_promary: 7.668 loss_dual: 0.975 loss_total: 8.642 batch_cost: 0.20638 sec reader_cost: 0.00030 sec ips: 38.76394 images/s eta: 0:06:07
    [05/29 18:50:00] ppgan.engine.trainer INFO: Iter: 18230/20000 lr: 9.992e-05 loss_promary: 8.297 loss_dual: 0.980 loss_total: 9.277 batch_cost: 0.21111 sec reader_cost: 0.00032 sec ips: 37.89482 images/s eta: 0:06:13
    [05/29 18:50:02] ppgan.engine.trainer INFO: Iter: 18240/20000 lr: 9.992e-05 loss_promary: 11.270 loss_dual: 1.213 loss_total: 12.483 batch_cost: 0.21218 sec reader_cost: 0.00033 sec ips: 37.70426 images/s eta: 0:06:13
    [05/29 18:50:04] ppgan.engine.trainer INFO: Iter: 18250/20000 lr: 9.992e-05 loss_promary: 9.077 loss_dual: 0.983 loss_total: 10.060 batch_cost: 0.20326 sec reader_cost: 0.00030 sec ips: 39.35844 images/s eta: 0:05:55
    [05/29 18:50:06] ppgan.engine.trainer INFO: Iter: 18260/20000 lr: 9.992e-05 loss_promary: 10.371 loss_dual: 1.110 loss_total: 11.480 batch_cost: 0.20571 sec reader_cost: 0.00031 sec ips: 38.88943 images/s eta: 0:05:57
    [05/29 18:50:09] ppgan.engine.trainer INFO: Iter: 18270/20000 lr: 9.992e-05 loss_promary: 10.073 loss_dual: 1.097 loss_total: 11.170 batch_cost: 0.20619 sec reader_cost: 0.00032 sec ips: 38.79945 images/s eta: 0:05:56
    [05/29 18:50:11] ppgan.engine.trainer INFO: Iter: 18280/20000 lr: 9.992e-05 loss_promary: 7.677 loss_dual: 0.968 loss_total: 8.645 batch_cost: 0.22080 sec reader_cost: 0.00030 sec ips: 36.23182 images/s eta: 0:06:19
    [05/29 18:50:13] ppgan.engine.trainer INFO: Iter: 18290/20000 lr: 9.992e-05 loss_promary: 10.184 loss_dual: 1.088 loss_total: 11.271 batch_cost: 0.20539 sec reader_cost: 0.00030 sec ips: 38.95091 images/s eta: 0:05:51
    [05/29 18:50:15] ppgan.engine.trainer INFO: Iter: 18300/20000 lr: 9.992e-05 loss_promary: 10.245 loss_dual: 1.152 loss_total: 11.396 batch_cost: 0.20487 sec reader_cost: 0.00031 sec ips: 39.04904 images/s eta: 0:05:48
    [05/29 18:50:17] ppgan.engine.trainer INFO: Iter: 18310/20000 lr: 9.992e-05 loss_promary: 9.616 loss_dual: 1.007 loss_total: 10.623 batch_cost: 0.20397 sec reader_cost: 0.00030 sec ips: 39.22092 images/s eta: 0:05:44
    [05/29 18:50:19] ppgan.engine.trainer INFO: Iter: 18320/20000 lr: 9.992e-05 loss_promary: 9.036 loss_dual: 1.030 loss_total: 10.066 batch_cost: 0.20538 sec reader_cost: 0.00030 sec ips: 38.95150 images/s eta: 0:05:45
    [05/29 18:50:21] ppgan.engine.trainer INFO: Iter: 18330/20000 lr: 9.992e-05 loss_promary: 9.282 loss_dual: 1.012 loss_total: 10.294 batch_cost: 0.20494 sec reader_cost: 0.00031 sec ips: 39.03641 images/s eta: 0:05:42
    [05/29 18:50:23] ppgan.engine.trainer INFO: Iter: 18340/20000 lr: 9.992e-05 loss_promary: 11.389 loss_dual: 1.203 loss_total: 12.592 batch_cost: 0.20417 sec reader_cost: 0.00031 sec ips: 39.18307 images/s eta: 0:05:38
    [05/29 18:50:25] ppgan.engine.trainer INFO: Iter: 18350/20000 lr: 9.992e-05 loss_promary: 11.489 loss_dual: 1.207 loss_total: 12.695 batch_cost: 0.20613 sec reader_cost: 0.00031 sec ips: 38.81058 images/s eta: 0:05:40
    [05/29 18:50:27] ppgan.engine.trainer INFO: Iter: 18360/20000 lr: 9.992e-05 loss_promary: 8.561 loss_dual: 0.984 loss_total: 9.545 batch_cost: 0.20387 sec reader_cost: 0.00031 sec ips: 39.23974 images/s eta: 0:05:34
    [05/29 18:50:29] ppgan.engine.trainer INFO: Iter: 18370/20000 lr: 9.992e-05 loss_promary: 11.339 loss_dual: 1.188 loss_total: 12.527 batch_cost: 0.20344 sec reader_cost: 0.00031 sec ips: 39.32269 images/s eta: 0:05:31
    [05/29 18:50:31] ppgan.engine.trainer INFO: Iter: 18380/20000 lr: 9.992e-05 loss_promary: 7.621 loss_dual: 0.941 loss_total: 8.562 batch_cost: 0.20613 sec reader_cost: 0.00030 sec ips: 38.80965 images/s eta: 0:05:33
    [05/29 18:50:33] ppgan.engine.trainer INFO: Iter: 18390/20000 lr: 9.992e-05 loss_promary: 10.941 loss_dual: 1.105 loss_total: 12.046 batch_cost: 0.20371 sec reader_cost: 0.00030 sec ips: 39.27099 images/s eta: 0:05:27
    [05/29 18:50:35] ppgan.engine.trainer INFO: Iter: 18400/20000 lr: 9.992e-05 loss_promary: 10.372 loss_dual: 1.159 loss_total: 11.531 batch_cost: 0.20542 sec reader_cost: 0.00030 sec ips: 38.94487 images/s eta: 0:05:28
    [05/29 18:50:37] ppgan.engine.trainer INFO: Iter: 18410/20000 lr: 9.992e-05 loss_promary: 10.418 loss_dual: 1.162 loss_total: 11.580 batch_cost: 0.20424 sec reader_cost: 0.00030 sec ips: 39.16883 images/s eta: 0:05:24
    [05/29 18:50:39] ppgan.engine.trainer INFO: Iter: 18420/20000 lr: 9.992e-05 loss_promary: 7.498 loss_dual: 0.931 loss_total: 8.428 batch_cost: 0.20617 sec reader_cost: 0.00033 sec ips: 38.80341 images/s eta: 0:05:25
    [05/29 18:50:42] ppgan.engine.trainer INFO: Iter: 18430/20000 lr: 9.992e-05 loss_promary: 9.069 loss_dual: 1.120 loss_total: 10.189 batch_cost: 0.23995 sec reader_cost: 0.00032 sec ips: 33.33986 images/s eta: 0:06:16
    [05/29 18:50:44] ppgan.engine.trainer INFO: Iter: 18440/20000 lr: 9.992e-05 loss_promary: 7.850 loss_dual: 1.062 loss_total: 8.911 batch_cost: 0.21277 sec reader_cost: 0.00031 sec ips: 37.60004 images/s eta: 0:05:31
    [05/29 18:50:46] ppgan.engine.trainer INFO: Iter: 18450/20000 lr: 9.992e-05 loss_promary: 10.063 loss_dual: 1.223 loss_total: 11.286 batch_cost: 0.20466 sec reader_cost: 0.00031 sec ips: 39.08994 images/s eta: 0:05:17
    [05/29 18:50:48] ppgan.engine.trainer INFO: Iter: 18460/20000 lr: 9.992e-05 loss_promary: 6.608 loss_dual: 0.860 loss_total: 7.468 batch_cost: 0.20429 sec reader_cost: 0.00031 sec ips: 39.15998 images/s eta: 0:05:14
    [05/29 18:50:50] ppgan.engine.trainer INFO: Iter: 18470/20000 lr: 9.992e-05 loss_promary: 7.653 loss_dual: 0.905 loss_total: 8.558 batch_cost: 0.20521 sec reader_cost: 0.00030 sec ips: 38.98355 images/s eta: 0:05:13
    [05/29 18:50:52] ppgan.engine.trainer INFO: Iter: 18480/20000 lr: 9.992e-05 loss_promary: 11.340 loss_dual: 1.232 loss_total: 12.572 batch_cost: 0.20375 sec reader_cost: 0.00027 sec ips: 39.26449 images/s eta: 0:05:09
    [05/29 18:50:55] ppgan.engine.trainer INFO: Iter: 18490/20000 lr: 9.992e-05 loss_promary: 8.466 loss_dual: 0.989 loss_total: 9.455 batch_cost: 0.26555 sec reader_cost: 0.04267 sec ips: 30.12563 images/s eta: 0:06:40
    [05/29 18:50:57] ppgan.engine.trainer INFO: Iter: 18500/20000 lr: 9.992e-05 loss_promary: 8.393 loss_dual: 0.973 loss_total: 9.366 batch_cost: 0.20553 sec reader_cost: 0.00031 sec ips: 38.92366 images/s eta: 0:05:08
    [05/29 18:50:59] ppgan.engine.trainer INFO: Iter: 18510/20000 lr: 9.992e-05 loss_promary: 12.875 loss_dual: 1.380 loss_total: 14.255 batch_cost: 0.20449 sec reader_cost: 0.00032 sec ips: 39.12219 images/s eta: 0:05:04
    [05/29 18:51:01] ppgan.engine.trainer INFO: Iter: 18520/20000 lr: 9.992e-05 loss_promary: 7.083 loss_dual: 0.873 loss_total: 7.956 batch_cost: 0.22493 sec reader_cost: 0.00032 sec ips: 35.56723 images/s eta: 0:05:32
    [05/29 18:51:04] ppgan.engine.trainer INFO: Iter: 18530/20000 lr: 9.992e-05 loss_promary: 10.441 loss_dual: 1.116 loss_total: 11.557 batch_cost: 0.23193 sec reader_cost: 0.00034 sec ips: 34.49365 images/s eta: 0:05:40
    [05/29 18:51:06] ppgan.engine.trainer INFO: Iter: 18540/20000 lr: 9.992e-05 loss_promary: 13.865 loss_dual: 1.443 loss_total: 15.308 batch_cost: 0.23317 sec reader_cost: 0.00035 sec ips: 34.31002 images/s eta: 0:05:40
    [05/29 18:51:08] ppgan.engine.trainer INFO: Iter: 18550/20000 lr: 9.992e-05 loss_promary: 6.532 loss_dual: 0.841 loss_total: 7.373 batch_cost: 0.23395 sec reader_cost: 0.00033 sec ips: 34.19546 images/s eta: 0:05:39
    [05/29 18:51:11] ppgan.engine.trainer INFO: Iter: 18560/20000 lr: 9.992e-05 loss_promary: 12.665 loss_dual: 1.358 loss_total: 14.023 batch_cost: 0.22801 sec reader_cost: 0.00031 sec ips: 35.08594 images/s eta: 0:05:28
    [05/29 18:51:13] ppgan.engine.trainer INFO: Iter: 18570/20000 lr: 9.992e-05 loss_promary: 8.715 loss_dual: 0.996 loss_total: 9.711 batch_cost: 0.23267 sec reader_cost: 0.00032 sec ips: 34.38338 images/s eta: 0:05:32
    [05/29 18:51:15] ppgan.engine.trainer INFO: Iter: 18580/20000 lr: 9.991e-05 loss_promary: 11.562 loss_dual: 1.226 loss_total: 12.788 batch_cost: 0.24661 sec reader_cost: 0.00032 sec ips: 32.43988 images/s eta: 0:05:50
    [05/29 18:51:18] ppgan.engine.trainer INFO: Iter: 18590/20000 lr: 9.991e-05 loss_promary: 8.185 loss_dual: 0.914 loss_total: 9.099 batch_cost: 0.22579 sec reader_cost: 0.00031 sec ips: 35.43072 images/s eta: 0:05:18
    [05/29 18:51:20] ppgan.engine.trainer INFO: Iter: 18600/20000 lr: 9.991e-05 loss_promary: 9.813 loss_dual: 1.046 loss_total: 10.858 batch_cost: 0.22871 sec reader_cost: 0.00031 sec ips: 34.97812 images/s eta: 0:05:20
    [05/29 18:51:22] ppgan.engine.trainer INFO: Iter: 18610/20000 lr: 9.991e-05 loss_promary: 7.984 loss_dual: 0.939 loss_total: 8.923 batch_cost: 0.23292 sec reader_cost: 0.00033 sec ips: 34.34589 images/s eta: 0:05:23
    [05/29 18:51:24] ppgan.engine.trainer INFO: Iter: 18620/20000 lr: 9.991e-05 loss_promary: 7.529 loss_dual: 0.950 loss_total: 8.479 batch_cost: 0.22871 sec reader_cost: 0.00032 sec ips: 34.97879 images/s eta: 0:05:15
    [05/29 18:51:27] ppgan.engine.trainer INFO: Iter: 18630/20000 lr: 9.991e-05 loss_promary: 6.562 loss_dual: 0.949 loss_total: 7.511 batch_cost: 0.22666 sec reader_cost: 0.00032 sec ips: 35.29574 images/s eta: 0:05:10
    [05/29 18:51:29] ppgan.engine.trainer INFO: Iter: 18640/20000 lr: 9.991e-05 loss_promary: 8.572 loss_dual: 1.000 loss_total: 9.572 batch_cost: 0.22941 sec reader_cost: 0.00031 sec ips: 34.87149 images/s eta: 0:05:12
    [05/29 18:51:31] ppgan.engine.trainer INFO: Iter: 18650/20000 lr: 9.991e-05 loss_promary: 9.555 loss_dual: 1.082 loss_total: 10.637 batch_cost: 0.20490 sec reader_cost: 0.00030 sec ips: 39.04302 images/s eta: 0:04:36
    [05/29 18:51:33] ppgan.engine.trainer INFO: Iter: 18660/20000 lr: 9.991e-05 loss_promary: 8.657 loss_dual: 1.010 loss_total: 9.667 batch_cost: 0.20316 sec reader_cost: 0.00030 sec ips: 39.37877 images/s eta: 0:04:32
    [05/29 18:51:35] ppgan.engine.trainer INFO: Iter: 18670/20000 lr: 9.991e-05 loss_promary: 8.992 loss_dual: 1.074 loss_total: 10.066 batch_cost: 0.20475 sec reader_cost: 0.00029 sec ips: 39.07134 images/s eta: 0:04:32
    [05/29 18:51:37] ppgan.engine.trainer INFO: Iter: 18680/20000 lr: 9.991e-05 loss_promary: 7.168 loss_dual: 0.947 loss_total: 8.115 batch_cost: 0.20564 sec reader_cost: 0.00030 sec ips: 38.90202 images/s eta: 0:04:31
    [05/29 18:51:39] ppgan.engine.trainer INFO: Iter: 18690/20000 lr: 9.991e-05 loss_promary: 7.756 loss_dual: 0.950 loss_total: 8.706 batch_cost: 0.20122 sec reader_cost: 0.00029 sec ips: 39.75840 images/s eta: 0:04:23
    [05/29 18:51:41] ppgan.engine.trainer INFO: Iter: 18700/20000 lr: 9.991e-05 loss_promary: 11.014 loss_dual: 1.271 loss_total: 12.285 batch_cost: 0.20212 sec reader_cost: 0.00029 sec ips: 39.57997 images/s eta: 0:04:22
    [05/29 18:51:43] ppgan.engine.trainer INFO: Iter: 18710/20000 lr: 9.991e-05 loss_promary: 7.232 loss_dual: 0.833 loss_total: 8.064 batch_cost: 0.20565 sec reader_cost: 0.00030 sec ips: 38.90117 images/s eta: 0:04:25
    [05/29 18:51:46] ppgan.engine.trainer INFO: Iter: 18720/20000 lr: 9.991e-05 loss_promary: 10.239 loss_dual: 1.113 loss_total: 11.352 batch_cost: 0.21732 sec reader_cost: 0.00030 sec ips: 36.81154 images/s eta: 0:04:38
    [05/29 18:51:48] ppgan.engine.trainer INFO: Iter: 18730/20000 lr: 9.991e-05 loss_promary: 11.737 loss_dual: 1.214 loss_total: 12.951 batch_cost: 0.20842 sec reader_cost: 0.00030 sec ips: 38.38417 images/s eta: 0:04:24
    [05/29 18:51:50] ppgan.engine.trainer INFO: Iter: 18740/20000 lr: 9.991e-05 loss_promary: 8.588 loss_dual: 0.961 loss_total: 9.549 batch_cost: 0.20115 sec reader_cost: 0.00029 sec ips: 39.77118 images/s eta: 0:04:13
    [05/29 18:51:52] ppgan.engine.trainer INFO: Iter: 18750/20000 lr: 9.991e-05 loss_promary: 7.826 loss_dual: 0.954 loss_total: 8.779 batch_cost: 0.20392 sec reader_cost: 0.00030 sec ips: 39.23160 images/s eta: 0:04:14
    [05/29 18:51:54] ppgan.engine.trainer INFO: Iter: 18760/20000 lr: 9.991e-05 loss_promary: 9.615 loss_dual: 1.028 loss_total: 10.642 batch_cost: 0.20363 sec reader_cost: 0.00031 sec ips: 39.28763 images/s eta: 0:04:12
    [05/29 18:51:56] ppgan.engine.trainer INFO: Iter: 18770/20000 lr: 9.991e-05 loss_promary: 15.735 loss_dual: 1.543 loss_total: 17.278 batch_cost: 0.20597 sec reader_cost: 0.00031 sec ips: 38.84133 images/s eta: 0:04:13
    [05/29 18:51:58] ppgan.engine.trainer INFO: Iter: 18780/20000 lr: 9.991e-05 loss_promary: 9.708 loss_dual: 1.068 loss_total: 10.776 batch_cost: 0.20540 sec reader_cost: 0.00030 sec ips: 38.94892 images/s eta: 0:04:10
    [05/29 18:52:00] ppgan.engine.trainer INFO: Iter: 18790/20000 lr: 9.991e-05 loss_promary: 13.851 loss_dual: 1.482 loss_total: 15.334 batch_cost: 0.20294 sec reader_cost: 0.00030 sec ips: 39.42029 images/s eta: 0:04:05
    [05/29 18:52:02] ppgan.engine.trainer INFO: Iter: 18800/20000 lr: 9.991e-05 loss_promary: 6.557 loss_dual: 0.771 loss_total: 7.328 batch_cost: 0.20822 sec reader_cost: 0.00030 sec ips: 38.42095 images/s eta: 0:04:09
    [05/29 18:52:04] ppgan.engine.trainer INFO: Iter: 18810/20000 lr: 9.991e-05 loss_promary: 8.821 loss_dual: 1.069 loss_total: 9.890 batch_cost: 0.20957 sec reader_cost: 0.00030 sec ips: 38.17317 images/s eta: 0:04:09
    [05/29 18:52:06] ppgan.engine.trainer INFO: Iter: 18820/20000 lr: 9.991e-05 loss_promary: 7.973 loss_dual: 0.934 loss_total: 8.907 batch_cost: 0.20941 sec reader_cost: 0.00032 sec ips: 38.20175 images/s eta: 0:04:07
    [05/29 18:52:08] ppgan.engine.trainer INFO: Iter: 18830/20000 lr: 9.991e-05 loss_promary: 13.531 loss_dual: 1.324 loss_total: 14.854 batch_cost: 0.20497 sec reader_cost: 0.00031 sec ips: 39.03034 images/s eta: 0:03:59
    [05/29 18:52:10] ppgan.engine.trainer INFO: Iter: 18840/20000 lr: 9.991e-05 loss_promary: 8.786 loss_dual: 1.027 loss_total: 9.813 batch_cost: 0.21574 sec reader_cost: 0.00032 sec ips: 37.08233 images/s eta: 0:04:10
    [05/29 18:52:12] ppgan.engine.trainer INFO: Iter: 18850/20000 lr: 9.991e-05 loss_promary: 9.565 loss_dual: 1.162 loss_total: 10.727 batch_cost: 0.20733 sec reader_cost: 0.00030 sec ips: 38.58596 images/s eta: 0:03:58
    [05/29 18:52:14] ppgan.engine.trainer INFO: Iter: 18860/20000 lr: 9.991e-05 loss_promary: 10.804 loss_dual: 1.248 loss_total: 12.051 batch_cost: 0.20553 sec reader_cost: 0.00030 sec ips: 38.92423 images/s eta: 0:03:54
    [05/29 18:52:17] ppgan.engine.trainer INFO: Iter: 18870/20000 lr: 9.991e-05 loss_promary: 9.901 loss_dual: 1.132 loss_total: 11.033 batch_cost: 0.20470 sec reader_cost: 0.00029 sec ips: 39.08218 images/s eta: 0:03:51
    [05/29 18:52:19] ppgan.engine.trainer INFO: Iter: 18880/20000 lr: 9.991e-05 loss_promary: 7.313 loss_dual: 0.951 loss_total: 8.264 batch_cost: 0.21296 sec reader_cost: 0.00030 sec ips: 37.56517 images/s eta: 0:03:58
    [05/29 18:52:21] ppgan.engine.trainer INFO: Iter: 18890/20000 lr: 9.991e-05 loss_promary: 6.498 loss_dual: 0.801 loss_total: 7.299 batch_cost: 0.20238 sec reader_cost: 0.00030 sec ips: 39.53055 images/s eta: 0:03:44
    [05/29 18:52:23] ppgan.engine.trainer INFO: Iter: 18900/20000 lr: 9.991e-05 loss_promary: 12.003 loss_dual: 1.287 loss_total: 13.290 batch_cost: 0.20209 sec reader_cost: 0.00030 sec ips: 39.58596 images/s eta: 0:03:42
    [05/29 18:52:25] ppgan.engine.trainer INFO: Iter: 18910/20000 lr: 9.991e-05 loss_promary: 7.742 loss_dual: 0.909 loss_total: 8.651 batch_cost: 0.20247 sec reader_cost: 0.00030 sec ips: 39.51111 images/s eta: 0:03:40
    [05/29 18:52:27] ppgan.engine.trainer INFO: Iter: 18920/20000 lr: 9.991e-05 loss_promary: 10.390 loss_dual: 1.106 loss_total: 11.496 batch_cost: 0.20348 sec reader_cost: 0.00030 sec ips: 39.31514 images/s eta: 0:03:39
    [05/29 18:52:29] ppgan.engine.trainer INFO: Iter: 18930/20000 lr: 9.991e-05 loss_promary: 9.565 loss_dual: 1.097 loss_total: 10.662 batch_cost: 0.20072 sec reader_cost: 0.00029 sec ips: 39.85710 images/s eta: 0:03:34
    [05/29 18:52:31] ppgan.engine.trainer INFO: Iter: 18940/20000 lr: 9.991e-05 loss_promary: 9.735 loss_dual: 1.100 loss_total: 10.835 batch_cost: 0.20607 sec reader_cost: 0.00030 sec ips: 38.82084 images/s eta: 0:03:38
    [05/29 18:52:33] ppgan.engine.trainer INFO: Iter: 18950/20000 lr: 9.991e-05 loss_promary: 7.567 loss_dual: 0.926 loss_total: 8.494 batch_cost: 0.20424 sec reader_cost: 0.00030 sec ips: 39.16957 images/s eta: 0:03:34
    [05/29 18:52:35] ppgan.engine.trainer INFO: Iter: 18960/20000 lr: 9.991e-05 loss_promary: 9.477 loss_dual: 1.090 loss_total: 10.567 batch_cost: 0.20218 sec reader_cost: 0.00029 sec ips: 39.56878 images/s eta: 0:03:30
    [05/29 18:52:37] ppgan.engine.trainer INFO: Iter: 18970/20000 lr: 9.991e-05 loss_promary: 6.211 loss_dual: 0.783 loss_total: 6.994 batch_cost: 0.20194 sec reader_cost: 0.00030 sec ips: 39.61647 images/s eta: 0:03:27
    [05/29 18:52:39] ppgan.engine.trainer INFO: Iter: 18980/20000 lr: 9.991e-05 loss_promary: 8.656 loss_dual: 0.967 loss_total: 9.623 batch_cost: 0.20206 sec reader_cost: 0.00029 sec ips: 39.59226 images/s eta: 0:03:26
    [05/29 18:52:41] ppgan.engine.trainer INFO: Iter: 18990/20000 lr: 9.991e-05 loss_promary: 9.264 loss_dual: 1.009 loss_total: 10.273 batch_cost: 0.20338 sec reader_cost: 0.00030 sec ips: 39.33493 images/s eta: 0:03:25
    [05/29 18:52:43] ppgan.engine.trainer INFO: Iter: 19000/20000 lr: 9.991e-05 loss_promary: 10.992 loss_dual: 1.151 loss_total: 12.142 batch_cost: 0.20329 sec reader_cost: 0.00030 sec ips: 39.35294 images/s eta: 0:03:23
    [05/29 18:52:43] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:52:45] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:52:47] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:52:48] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:52:50] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:52:52] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:52:54] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:52:55] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:52:57] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:52:59] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:53:01] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:53:02] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:53:04] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:53:06] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:53:08] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:53:09] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:53:11] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:53:13] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:53:14] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:53:16] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:53:18] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:53:20] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:53:21] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:53:23] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:53:25] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:53:27] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:53:28] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:53:30] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:53:32] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:53:33] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:53:35] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:53:37] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:53:39] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:53:40] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:53:42] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:53:44] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:53:45] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:53:47] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:53:49] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:53:51] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:53:52] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:53:54] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:53:56] ppgan.engine.trainer INFO: Metric psnr: 29.4229
    [05/29 18:53:56] ppgan.engine.trainer INFO: Metric ssim: 0.8016
    [05/29 18:53:58] ppgan.engine.trainer INFO: Iter: 19010/20000 lr: 9.991e-05 loss_promary: 6.878 loss_dual: 0.798 loss_total: 7.676 batch_cost: 0.20772 sec reader_cost: 0.00037 sec ips: 38.51312 images/s eta: 0:03:25
    [05/29 18:54:00] ppgan.engine.trainer INFO: Iter: 19020/20000 lr: 9.991e-05 loss_promary: 7.282 loss_dual: 0.847 loss_total: 8.129 batch_cost: 0.20460 sec reader_cost: 0.00030 sec ips: 39.09979 images/s eta: 0:03:20
    [05/29 18:54:02] ppgan.engine.trainer INFO: Iter: 19030/20000 lr: 9.991e-05 loss_promary: 11.177 loss_dual: 1.196 loss_total: 12.373 batch_cost: 0.20423 sec reader_cost: 0.00031 sec ips: 39.17091 images/s eta: 0:03:18
    [05/29 18:54:04] ppgan.engine.trainer INFO: Iter: 19040/20000 lr: 9.991e-05 loss_promary: 9.340 loss_dual: 0.998 loss_total: 10.339 batch_cost: 0.20160 sec reader_cost: 0.00030 sec ips: 39.68234 images/s eta: 0:03:13
    [05/29 18:54:06] ppgan.engine.trainer INFO: Iter: 19050/20000 lr: 9.991e-05 loss_promary: 9.817 loss_dual: 1.076 loss_total: 10.893 batch_cost: 0.20222 sec reader_cost: 0.00030 sec ips: 39.56127 images/s eta: 0:03:12
    [05/29 18:54:08] ppgan.engine.trainer INFO: Iter: 19060/20000 lr: 9.991e-05 loss_promary: 6.490 loss_dual: 0.805 loss_total: 7.295 batch_cost: 0.20248 sec reader_cost: 0.00030 sec ips: 39.50913 images/s eta: 0:03:10
    [05/29 18:54:10] ppgan.engine.trainer INFO: Iter: 19070/20000 lr: 9.991e-05 loss_promary: 9.791 loss_dual: 1.033 loss_total: 10.824 batch_cost: 0.20266 sec reader_cost: 0.00030 sec ips: 39.47494 images/s eta: 0:03:08
    [05/29 18:54:12] ppgan.engine.trainer INFO: Iter: 19080/20000 lr: 9.991e-05 loss_promary: 7.276 loss_dual: 0.912 loss_total: 8.188 batch_cost: 0.20192 sec reader_cost: 0.00030 sec ips: 39.61950 images/s eta: 0:03:05
    [05/29 18:54:14] ppgan.engine.trainer INFO: Iter: 19090/20000 lr: 9.991e-05 loss_promary: 8.611 loss_dual: 1.073 loss_total: 9.684 batch_cost: 0.20184 sec reader_cost: 0.00030 sec ips: 39.63629 images/s eta: 0:03:03
    [05/29 18:54:16] ppgan.engine.trainer INFO: Iter: 19100/20000 lr: 9.991e-05 loss_promary: 11.359 loss_dual: 1.195 loss_total: 12.554 batch_cost: 0.20258 sec reader_cost: 0.00030 sec ips: 39.49044 images/s eta: 0:03:02
    [05/29 18:54:18] ppgan.engine.trainer INFO: Iter: 19110/20000 lr: 9.991e-05 loss_promary: 6.927 loss_dual: 0.800 loss_total: 7.726 batch_cost: 0.20311 sec reader_cost: 0.00030 sec ips: 39.38844 images/s eta: 0:03:00
    [05/29 18:54:21] ppgan.engine.trainer INFO: Iter: 19120/20000 lr: 9.991e-05 loss_promary: 7.656 loss_dual: 0.879 loss_total: 8.534 batch_cost: 0.20530 sec reader_cost: 0.00031 sec ips: 38.96821 images/s eta: 0:03:00
    [05/29 18:54:23] ppgan.engine.trainer INFO: Iter: 19130/20000 lr: 9.991e-05 loss_promary: 9.415 loss_dual: 1.056 loss_total: 10.471 batch_cost: 0.21511 sec reader_cost: 0.00033 sec ips: 37.19105 images/s eta: 0:03:07
    [05/29 18:54:25] ppgan.engine.trainer INFO: Iter: 19140/20000 lr: 9.991e-05 loss_promary: 11.345 loss_dual: 1.206 loss_total: 12.550 batch_cost: 0.22454 sec reader_cost: 0.00032 sec ips: 35.62828 images/s eta: 0:03:13
    [05/29 18:54:27] ppgan.engine.trainer INFO: Iter: 19150/20000 lr: 9.991e-05 loss_promary: 7.613 loss_dual: 0.876 loss_total: 8.489 batch_cost: 0.20839 sec reader_cost: 0.00032 sec ips: 38.39004 images/s eta: 0:02:57
    [05/29 18:54:29] ppgan.engine.trainer INFO: Iter: 19160/20000 lr: 9.991e-05 loss_promary: 6.608 loss_dual: 0.811 loss_total: 7.419 batch_cost: 0.20451 sec reader_cost: 0.00032 sec ips: 39.11746 images/s eta: 0:02:51
    [05/29 18:54:31] ppgan.engine.trainer INFO: Iter: 19170/20000 lr: 9.991e-05 loss_promary: 8.253 loss_dual: 0.957 loss_total: 9.210 batch_cost: 0.20507 sec reader_cost: 0.00031 sec ips: 39.01139 images/s eta: 0:02:50
    [05/29 18:54:33] ppgan.engine.trainer INFO: Iter: 19180/20000 lr: 9.991e-05 loss_promary: 6.966 loss_dual: 0.868 loss_total: 7.833 batch_cost: 0.20566 sec reader_cost: 0.00032 sec ips: 38.89969 images/s eta: 0:02:48
    [05/29 18:54:35] ppgan.engine.trainer INFO: Iter: 19190/20000 lr: 9.991e-05 loss_promary: 8.045 loss_dual: 0.969 loss_total: 9.015 batch_cost: 0.20576 sec reader_cost: 0.00032 sec ips: 38.87981 images/s eta: 0:02:46
    [05/29 18:54:37] ppgan.engine.trainer INFO: Iter: 19200/20000 lr: 9.991e-05 loss_promary: 10.741 loss_dual: 1.273 loss_total: 12.014 batch_cost: 0.20428 sec reader_cost: 0.00031 sec ips: 39.16201 images/s eta: 0:02:43
    [05/29 18:54:39] ppgan.engine.trainer INFO: Iter: 19210/20000 lr: 9.991e-05 loss_promary: 8.437 loss_dual: 1.069 loss_total: 9.506 batch_cost: 0.20521 sec reader_cost: 0.00032 sec ips: 38.98384 images/s eta: 0:02:42
    [05/29 18:54:41] ppgan.engine.trainer INFO: Iter: 19220/20000 lr: 9.991e-05 loss_promary: 8.885 loss_dual: 1.075 loss_total: 9.959 batch_cost: 0.20439 sec reader_cost: 0.00031 sec ips: 39.14041 images/s eta: 0:02:39
    [05/29 18:54:43] ppgan.engine.trainer INFO: Iter: 19230/20000 lr: 9.991e-05 loss_promary: 7.930 loss_dual: 0.962 loss_total: 8.892 batch_cost: 0.20275 sec reader_cost: 0.00030 sec ips: 39.45684 images/s eta: 0:02:36
    [05/29 18:54:45] ppgan.engine.trainer INFO: Iter: 19240/20000 lr: 9.991e-05 loss_promary: 6.123 loss_dual: 0.788 loss_total: 6.912 batch_cost: 0.20242 sec reader_cost: 0.00031 sec ips: 39.52158 images/s eta: 0:02:33
    [05/29 18:54:48] ppgan.engine.trainer INFO: Iter: 19250/20000 lr: 9.991e-05 loss_promary: 7.130 loss_dual: 0.886 loss_total: 8.016 batch_cost: 0.21287 sec reader_cost: 0.00031 sec ips: 37.58186 images/s eta: 0:02:39
    [05/29 18:54:50] ppgan.engine.trainer INFO: Iter: 19260/20000 lr: 9.991e-05 loss_promary: 9.936 loss_dual: 1.232 loss_total: 11.168 batch_cost: 0.20760 sec reader_cost: 0.00030 sec ips: 38.53501 images/s eta: 0:02:33
    [05/29 18:54:52] ppgan.engine.trainer INFO: Iter: 19270/20000 lr: 9.991e-05 loss_promary: 9.497 loss_dual: 1.123 loss_total: 10.620 batch_cost: 0.20486 sec reader_cost: 0.00030 sec ips: 39.05036 images/s eta: 0:02:29
    [05/29 18:54:54] ppgan.engine.trainer INFO: Iter: 19280/20000 lr: 9.991e-05 loss_promary: 7.483 loss_dual: 0.909 loss_total: 8.392 batch_cost: 0.20490 sec reader_cost: 0.00030 sec ips: 39.04428 images/s eta: 0:02:27
    [05/29 18:54:56] ppgan.engine.trainer INFO: Iter: 19290/20000 lr: 9.991e-05 loss_promary: 9.299 loss_dual: 1.107 loss_total: 10.407 batch_cost: 0.21506 sec reader_cost: 0.00031 sec ips: 37.19820 images/s eta: 0:02:32
    [05/29 18:54:58] ppgan.engine.trainer INFO: Iter: 19300/20000 lr: 9.991e-05 loss_promary: 12.067 loss_dual: 1.351 loss_total: 13.417 batch_cost: 0.21173 sec reader_cost: 0.00029 sec ips: 37.78400 images/s eta: 0:02:28
    [05/29 18:55:00] ppgan.engine.trainer INFO: Iter: 19310/20000 lr: 9.991e-05 loss_promary: 8.370 loss_dual: 0.943 loss_total: 9.313 batch_cost: 0.20412 sec reader_cost: 0.00029 sec ips: 39.19305 images/s eta: 0:02:20
    [05/29 18:55:02] ppgan.engine.trainer INFO: Iter: 19320/20000 lr: 9.991e-05 loss_promary: 13.889 loss_dual: 1.438 loss_total: 15.327 batch_cost: 0.20018 sec reader_cost: 0.00026 sec ips: 39.96417 images/s eta: 0:02:16
    [05/29 18:55:05] ppgan.engine.trainer INFO: Iter: 19330/20000 lr: 9.991e-05 loss_promary: 9.134 loss_dual: 1.058 loss_total: 10.192 batch_cost: 0.25442 sec reader_cost: 0.03784 sec ips: 31.44384 images/s eta: 0:02:50
    [05/29 18:55:07] ppgan.engine.trainer INFO: Iter: 19340/20000 lr: 9.991e-05 loss_promary: 9.319 loss_dual: 1.013 loss_total: 10.332 batch_cost: 0.20163 sec reader_cost: 0.00029 sec ips: 39.67684 images/s eta: 0:02:13
    [05/29 18:55:09] ppgan.engine.trainer INFO: Iter: 19350/20000 lr: 9.991e-05 loss_promary: 9.112 loss_dual: 1.050 loss_total: 10.163 batch_cost: 0.20047 sec reader_cost: 0.00029 sec ips: 39.90576 images/s eta: 0:02:10
    [05/29 18:55:11] ppgan.engine.trainer INFO: Iter: 19360/20000 lr: 9.991e-05 loss_promary: 9.304 loss_dual: 1.068 loss_total: 10.372 batch_cost: 0.20153 sec reader_cost: 0.00029 sec ips: 39.69718 images/s eta: 0:02:08
    [05/29 18:55:13] ppgan.engine.trainer INFO: Iter: 19370/20000 lr: 9.991e-05 loss_promary: 8.099 loss_dual: 0.990 loss_total: 9.090 batch_cost: 0.20129 sec reader_cost: 0.00029 sec ips: 39.74378 images/s eta: 0:02:06
    [05/29 18:55:15] ppgan.engine.trainer INFO: Iter: 19380/20000 lr: 9.991e-05 loss_promary: 9.964 loss_dual: 1.089 loss_total: 11.053 batch_cost: 0.20082 sec reader_cost: 0.00030 sec ips: 39.83601 images/s eta: 0:02:04
    [05/29 18:55:17] ppgan.engine.trainer INFO: Iter: 19390/20000 lr: 9.991e-05 loss_promary: 14.064 loss_dual: 1.599 loss_total: 15.663 batch_cost: 0.19997 sec reader_cost: 0.00028 sec ips: 40.00567 images/s eta: 0:02:01
    [05/29 18:55:19] ppgan.engine.trainer INFO: Iter: 19400/20000 lr: 9.991e-05 loss_promary: 10.063 loss_dual: 1.104 loss_total: 11.166 batch_cost: 0.19970 sec reader_cost: 0.00029 sec ips: 40.06014 images/s eta: 0:01:59
    [05/29 18:55:21] ppgan.engine.trainer INFO: Iter: 19410/20000 lr: 9.991e-05 loss_promary: 9.840 loss_dual: 1.106 loss_total: 10.946 batch_cost: 0.20276 sec reader_cost: 0.00029 sec ips: 39.45640 images/s eta: 0:01:59
    [05/29 18:55:23] ppgan.engine.trainer INFO: Iter: 19420/20000 lr: 9.991e-05 loss_promary: 10.795 loss_dual: 1.230 loss_total: 12.024 batch_cost: 0.20151 sec reader_cost: 0.00029 sec ips: 39.69953 images/s eta: 0:01:56
    [05/29 18:55:25] ppgan.engine.trainer INFO: Iter: 19430/20000 lr: 9.991e-05 loss_promary: 13.627 loss_dual: 1.434 loss_total: 15.061 batch_cost: 0.20195 sec reader_cost: 0.00028 sec ips: 39.61402 images/s eta: 0:01:55
    [05/29 18:55:27] ppgan.engine.trainer INFO: Iter: 19440/20000 lr: 9.991e-05 loss_promary: 8.955 loss_dual: 1.025 loss_total: 9.980 batch_cost: 0.21335 sec reader_cost: 0.00030 sec ips: 37.49672 images/s eta: 0:01:59
    [05/29 18:55:29] ppgan.engine.trainer INFO: Iter: 19450/20000 lr: 9.991e-05 loss_promary: 10.097 loss_dual: 1.142 loss_total: 11.239 batch_cost: 0.23330 sec reader_cost: 0.00032 sec ips: 34.28997 images/s eta: 0:02:08
    [05/29 18:55:31] ppgan.engine.trainer INFO: Iter: 19460/20000 lr: 9.991e-05 loss_promary: 9.714 loss_dual: 1.078 loss_total: 10.793 batch_cost: 0.20466 sec reader_cost: 0.00030 sec ips: 39.08917 images/s eta: 0:01:50
    [05/29 18:55:33] ppgan.engine.trainer INFO: Iter: 19470/20000 lr: 9.991e-05 loss_promary: 7.443 loss_dual: 0.892 loss_total: 8.334 batch_cost: 0.20713 sec reader_cost: 0.00031 sec ips: 38.62255 images/s eta: 0:01:49
    [05/29 18:55:35] ppgan.engine.trainer INFO: Iter: 19480/20000 lr: 9.991e-05 loss_promary: 10.036 loss_dual: 1.100 loss_total: 11.136 batch_cost: 0.21008 sec reader_cost: 0.00032 sec ips: 38.08088 images/s eta: 0:01:49
    [05/29 18:55:38] ppgan.engine.trainer INFO: Iter: 19490/20000 lr: 9.991e-05 loss_promary: 9.712 loss_dual: 1.101 loss_total: 10.814 batch_cost: 0.20288 sec reader_cost: 0.00030 sec ips: 39.43252 images/s eta: 0:01:43
    [05/29 18:55:40] ppgan.engine.trainer INFO: Iter: 19500/20000 lr: 9.991e-05 loss_promary: 10.152 loss_dual: 1.099 loss_total: 11.251 batch_cost: 0.20147 sec reader_cost: 0.00029 sec ips: 39.70753 images/s eta: 0:01:40
    [05/29 18:55:42] ppgan.engine.trainer INFO: Iter: 19510/20000 lr: 9.991e-05 loss_promary: 11.682 loss_dual: 1.283 loss_total: 12.965 batch_cost: 0.20324 sec reader_cost: 0.00032 sec ips: 39.36167 images/s eta: 0:01:39
    [05/29 18:55:44] ppgan.engine.trainer INFO: Iter: 19520/20000 lr: 9.991e-05 loss_promary: 6.915 loss_dual: 0.869 loss_total: 7.784 batch_cost: 0.20219 sec reader_cost: 0.00030 sec ips: 39.56662 images/s eta: 0:01:37
    [05/29 18:55:46] ppgan.engine.trainer INFO: Iter: 19530/20000 lr: 9.991e-05 loss_promary: 14.782 loss_dual: 1.543 loss_total: 16.326 batch_cost: 0.20267 sec reader_cost: 0.00030 sec ips: 39.47313 images/s eta: 0:01:35
    [05/29 18:55:48] ppgan.engine.trainer INFO: Iter: 19540/20000 lr: 9.991e-05 loss_promary: 8.919 loss_dual: 1.077 loss_total: 9.997 batch_cost: 0.20744 sec reader_cost: 0.00030 sec ips: 38.56562 images/s eta: 0:01:35
    [05/29 18:55:50] ppgan.engine.trainer INFO: Iter: 19550/20000 lr: 9.991e-05 loss_promary: 10.052 loss_dual: 1.111 loss_total: 11.163 batch_cost: 0.20102 sec reader_cost: 0.00030 sec ips: 39.79692 images/s eta: 0:01:30
    [05/29 18:55:52] ppgan.engine.trainer INFO: Iter: 19560/20000 lr: 9.991e-05 loss_promary: 8.695 loss_dual: 0.991 loss_total: 9.686 batch_cost: 0.20195 sec reader_cost: 0.00029 sec ips: 39.61322 images/s eta: 0:01:28
    [05/29 18:55:54] ppgan.engine.trainer INFO: Iter: 19570/20000 lr: 9.991e-05 loss_promary: 8.873 loss_dual: 0.997 loss_total: 9.870 batch_cost: 0.21454 sec reader_cost: 0.00032 sec ips: 37.28822 images/s eta: 0:01:32
    [05/29 18:55:56] ppgan.engine.trainer INFO: Iter: 19580/20000 lr: 9.991e-05 loss_promary: 11.503 loss_dual: 1.233 loss_total: 12.736 batch_cost: 0.21024 sec reader_cost: 0.00032 sec ips: 38.05117 images/s eta: 0:01:28
    [05/29 18:55:58] ppgan.engine.trainer INFO: Iter: 19590/20000 lr: 9.991e-05 loss_promary: 7.839 loss_dual: 0.982 loss_total: 8.820 batch_cost: 0.21307 sec reader_cost: 0.00032 sec ips: 37.54665 images/s eta: 0:01:27
    [05/29 18:56:00] ppgan.engine.trainer INFO: Iter: 19600/20000 lr: 9.991e-05 loss_promary: 6.748 loss_dual: 0.815 loss_total: 7.563 batch_cost: 0.22144 sec reader_cost: 0.00033 sec ips: 36.12734 images/s eta: 0:01:28
    [05/29 18:56:02] ppgan.engine.trainer INFO: Iter: 19610/20000 lr: 9.991e-05 loss_promary: 10.130 loss_dual: 1.133 loss_total: 11.264 batch_cost: 0.20440 sec reader_cost: 0.00035 sec ips: 39.13879 images/s eta: 0:01:19
    [05/29 18:56:04] ppgan.engine.trainer INFO: Iter: 19620/20000 lr: 9.991e-05 loss_promary: 11.786 loss_dual: 1.333 loss_total: 13.119 batch_cost: 0.20159 sec reader_cost: 0.00031 sec ips: 39.68536 images/s eta: 0:01:16
    [05/29 18:56:06] ppgan.engine.trainer INFO: Iter: 19630/20000 lr: 9.991e-05 loss_promary: 7.901 loss_dual: 0.979 loss_total: 8.881 batch_cost: 0.20159 sec reader_cost: 0.00029 sec ips: 39.68473 images/s eta: 0:01:14
    [05/29 18:56:08] ppgan.engine.trainer INFO: Iter: 19640/20000 lr: 9.990e-05 loss_promary: 9.466 loss_dual: 1.072 loss_total: 10.539 batch_cost: 0.20239 sec reader_cost: 0.00029 sec ips: 39.52855 images/s eta: 0:01:12
    [05/29 18:56:11] ppgan.engine.trainer INFO: Iter: 19650/20000 lr: 9.990e-05 loss_promary: 8.569 loss_dual: 0.915 loss_total: 9.484 batch_cost: 0.20376 sec reader_cost: 0.00030 sec ips: 39.26143 images/s eta: 0:01:11
    [05/29 18:56:13] ppgan.engine.trainer INFO: Iter: 19660/20000 lr: 9.990e-05 loss_promary: 12.151 loss_dual: 1.244 loss_total: 13.396 batch_cost: 0.20378 sec reader_cost: 0.00029 sec ips: 39.25746 images/s eta: 0:01:09
    [05/29 18:56:15] ppgan.engine.trainer INFO: Iter: 19670/20000 lr: 9.990e-05 loss_promary: 8.115 loss_dual: 0.936 loss_total: 9.051 batch_cost: 0.20363 sec reader_cost: 0.00030 sec ips: 39.28710 images/s eta: 0:01:07
    [05/29 18:56:17] ppgan.engine.trainer INFO: Iter: 19680/20000 lr: 9.990e-05 loss_promary: 14.364 loss_dual: 1.447 loss_total: 15.810 batch_cost: 0.20066 sec reader_cost: 0.00029 sec ips: 39.86872 images/s eta: 0:01:04
    [05/29 18:56:19] ppgan.engine.trainer INFO: Iter: 19690/20000 lr: 9.990e-05 loss_promary: 10.884 loss_dual: 1.130 loss_total: 12.014 batch_cost: 0.19901 sec reader_cost: 0.00028 sec ips: 40.19934 images/s eta: 0:01:01
    [05/29 18:56:21] ppgan.engine.trainer INFO: Iter: 19700/20000 lr: 9.990e-05 loss_promary: 8.281 loss_dual: 0.918 loss_total: 9.199 batch_cost: 0.20134 sec reader_cost: 0.00029 sec ips: 39.73435 images/s eta: 0:01:00
    [05/29 18:56:23] ppgan.engine.trainer INFO: Iter: 19710/20000 lr: 9.990e-05 loss_promary: 8.426 loss_dual: 0.950 loss_total: 9.377 batch_cost: 0.20130 sec reader_cost: 0.00029 sec ips: 39.74086 images/s eta: 0:00:58
    [05/29 18:56:25] ppgan.engine.trainer INFO: Iter: 19720/20000 lr: 9.990e-05 loss_promary: 8.487 loss_dual: 0.995 loss_total: 9.481 batch_cost: 0.20082 sec reader_cost: 0.00029 sec ips: 39.83681 images/s eta: 0:00:56
    [05/29 18:56:27] ppgan.engine.trainer INFO: Iter: 19730/20000 lr: 9.990e-05 loss_promary: 9.620 loss_dual: 1.043 loss_total: 10.663 batch_cost: 0.19960 sec reader_cost: 0.00029 sec ips: 40.08033 images/s eta: 0:00:53
    [05/29 18:56:29] ppgan.engine.trainer INFO: Iter: 19740/20000 lr: 9.990e-05 loss_promary: 7.924 loss_dual: 0.879 loss_total: 8.803 batch_cost: 0.20060 sec reader_cost: 0.00029 sec ips: 39.87946 images/s eta: 0:00:52
    [05/29 18:56:31] ppgan.engine.trainer INFO: Iter: 19750/20000 lr: 9.990e-05 loss_promary: 7.932 loss_dual: 0.884 loss_total: 8.815 batch_cost: 0.20521 sec reader_cost: 0.00030 sec ips: 38.98442 images/s eta: 0:00:51
    [05/29 18:56:33] ppgan.engine.trainer INFO: Iter: 19760/20000 lr: 9.990e-05 loss_promary: 9.154 loss_dual: 1.060 loss_total: 10.214 batch_cost: 0.23580 sec reader_cost: 0.00031 sec ips: 33.92773 images/s eta: 0:00:56
    [05/29 18:56:35] ppgan.engine.trainer INFO: Iter: 19770/20000 lr: 9.990e-05 loss_promary: 7.906 loss_dual: 0.938 loss_total: 8.844 batch_cost: 0.20467 sec reader_cost: 0.00030 sec ips: 39.08806 images/s eta: 0:00:47
    [05/29 18:56:37] ppgan.engine.trainer INFO: Iter: 19780/20000 lr: 9.990e-05 loss_promary: 7.820 loss_dual: 0.947 loss_total: 8.767 batch_cost: 0.20128 sec reader_cost: 0.00029 sec ips: 39.74617 images/s eta: 0:00:44
    [05/29 18:56:39] ppgan.engine.trainer INFO: Iter: 19790/20000 lr: 9.990e-05 loss_promary: 8.508 loss_dual: 0.983 loss_total: 9.491 batch_cost: 0.20082 sec reader_cost: 0.00030 sec ips: 39.83629 images/s eta: 0:00:42
    [05/29 18:56:41] ppgan.engine.trainer INFO: Iter: 19800/20000 lr: 9.990e-05 loss_promary: 7.872 loss_dual: 0.903 loss_total: 8.775 batch_cost: 0.20135 sec reader_cost: 0.00030 sec ips: 39.73237 images/s eta: 0:00:40
    [05/29 18:56:43] ppgan.engine.trainer INFO: Iter: 19810/20000 lr: 9.990e-05 loss_promary: 6.087 loss_dual: 0.753 loss_total: 6.840 batch_cost: 0.20027 sec reader_cost: 0.00029 sec ips: 39.94699 images/s eta: 0:00:38
    [05/29 18:56:45] ppgan.engine.trainer INFO: Iter: 19820/20000 lr: 9.990e-05 loss_promary: 10.868 loss_dual: 1.290 loss_total: 12.159 batch_cost: 0.19945 sec reader_cost: 0.00028 sec ips: 40.11109 images/s eta: 0:00:35
    [05/29 18:56:47] ppgan.engine.trainer INFO: Iter: 19830/20000 lr: 9.990e-05 loss_promary: 6.593 loss_dual: 0.867 loss_total: 7.460 batch_cost: 0.19882 sec reader_cost: 0.00029 sec ips: 40.23645 images/s eta: 0:00:33
    [05/29 18:56:49] ppgan.engine.trainer INFO: Iter: 19840/20000 lr: 9.990e-05 loss_promary: 8.825 loss_dual: 1.044 loss_total: 9.868 batch_cost: 0.20826 sec reader_cost: 0.00029 sec ips: 38.41293 images/s eta: 0:00:33
    [05/29 18:56:51] ppgan.engine.trainer INFO: Iter: 19850/20000 lr: 9.990e-05 loss_promary: 6.902 loss_dual: 0.784 loss_total: 7.686 batch_cost: 0.20555 sec reader_cost: 0.00030 sec ips: 38.92026 images/s eta: 0:00:30
    [05/29 18:56:53] ppgan.engine.trainer INFO: Iter: 19860/20000 lr: 9.990e-05 loss_promary: 8.091 loss_dual: 0.932 loss_total: 9.023 batch_cost: 0.20358 sec reader_cost: 0.00029 sec ips: 39.29570 images/s eta: 0:00:28
    [05/29 18:56:55] ppgan.engine.trainer INFO: Iter: 19870/20000 lr: 9.990e-05 loss_promary: 9.349 loss_dual: 1.053 loss_total: 10.402 batch_cost: 0.20606 sec reader_cost: 0.00032 sec ips: 38.82360 images/s eta: 0:00:26
    [05/29 18:56:57] ppgan.engine.trainer INFO: Iter: 19880/20000 lr: 9.990e-05 loss_promary: 8.408 loss_dual: 0.954 loss_total: 9.362 batch_cost: 0.20465 sec reader_cost: 0.00030 sec ips: 39.09138 images/s eta: 0:00:24
    [05/29 18:56:59] ppgan.engine.trainer INFO: Iter: 19890/20000 lr: 9.990e-05 loss_promary: 9.830 loss_dual: 1.168 loss_total: 10.999 batch_cost: 0.20324 sec reader_cost: 0.00030 sec ips: 39.36155 images/s eta: 0:00:22
    [05/29 18:57:02] ppgan.engine.trainer INFO: Iter: 19900/20000 lr: 9.990e-05 loss_promary: 7.590 loss_dual: 0.877 loss_total: 8.467 batch_cost: 0.20522 sec reader_cost: 0.00030 sec ips: 38.98199 images/s eta: 0:00:20
    [05/29 18:57:04] ppgan.engine.trainer INFO: Iter: 19910/20000 lr: 9.990e-05 loss_promary: 11.218 loss_dual: 1.188 loss_total: 12.406 batch_cost: 0.21352 sec reader_cost: 0.00030 sec ips: 37.46644 images/s eta: 0:00:19
    [05/29 18:57:06] ppgan.engine.trainer INFO: Iter: 19920/20000 lr: 9.990e-05 loss_promary: 8.219 loss_dual: 0.984 loss_total: 9.203 batch_cost: 0.20869 sec reader_cost: 0.00030 sec ips: 38.33524 images/s eta: 0:00:16
    [05/29 18:57:08] ppgan.engine.trainer INFO: Iter: 19930/20000 lr: 9.990e-05 loss_promary: 7.666 loss_dual: 0.877 loss_total: 8.543 batch_cost: 0.20001 sec reader_cost: 0.00029 sec ips: 39.99859 images/s eta: 0:00:14
    [05/29 18:57:10] ppgan.engine.trainer INFO: Iter: 19940/20000 lr: 9.990e-05 loss_promary: 8.462 loss_dual: 0.967 loss_total: 9.430 batch_cost: 0.20230 sec reader_cost: 0.00030 sec ips: 39.54459 images/s eta: 0:00:12
    [05/29 18:57:12] ppgan.engine.trainer INFO: Iter: 19950/20000 lr: 9.990e-05 loss_promary: 10.219 loss_dual: 1.161 loss_total: 11.380 batch_cost: 0.20357 sec reader_cost: 0.00032 sec ips: 39.29935 images/s eta: 0:00:10
    [05/29 18:57:14] ppgan.engine.trainer INFO: Iter: 19960/20000 lr: 9.990e-05 loss_promary: 10.733 loss_dual: 1.337 loss_total: 12.069 batch_cost: 0.20929 sec reader_cost: 0.00030 sec ips: 38.22524 images/s eta: 0:00:08
    [05/29 18:57:16] ppgan.engine.trainer INFO: Iter: 19970/20000 lr: 9.990e-05 loss_promary: 10.525 loss_dual: 1.171 loss_total: 11.695 batch_cost: 0.20607 sec reader_cost: 0.00032 sec ips: 38.82246 images/s eta: 0:00:06
    [05/29 18:57:18] ppgan.engine.trainer INFO: Iter: 19980/20000 lr: 9.990e-05 loss_promary: 7.848 loss_dual: 0.932 loss_total: 8.781 batch_cost: 0.20411 sec reader_cost: 0.00031 sec ips: 39.19393 images/s eta: 0:00:04
    [05/29 18:57:20] ppgan.engine.trainer INFO: Iter: 19990/20000 lr: 9.990e-05 loss_promary: 10.081 loss_dual: 1.056 loss_total: 11.137 batch_cost: 0.20616 sec reader_cost: 0.00030 sec ips: 38.80536 images/s eta: 0:00:02
    [05/29 18:57:22] ppgan.engine.trainer INFO: Iter: 20000/20000 lr: 9.990e-05 loss_promary: 8.080 loss_dual: 0.925 loss_total: 9.005 batch_cost: 0.20322 sec reader_cost: 0.00030 sec ips: 39.36626 images/s eta: 0:00:00
    [05/29 18:57:22] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/29 18:57:24] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/29 18:57:26] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/29 18:57:27] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/29 18:57:29] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/29 18:57:31] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/29 18:57:32] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/29 18:57:34] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/29 18:57:36] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/29 18:57:38] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/29 18:57:39] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/29 18:57:41] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/29 18:57:43] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/29 18:57:45] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/29 18:57:46] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/29 18:57:48] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/29 18:57:50] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/29 18:57:52] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/29 18:57:53] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/29 18:57:55] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/29 18:57:57] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/29 18:57:58] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/29 18:58:00] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/29 18:58:02] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/29 18:58:04] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/29 18:58:05] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/29 18:58:07] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/29 18:58:09] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/29 18:58:11] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/29 18:58:12] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/29 18:58:14] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/29 18:58:16] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/29 18:58:17] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/29 18:58:19] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/29 18:58:21] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/29 18:58:23] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/29 18:58:24] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/29 18:58:26] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/29 18:58:28] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/29 18:58:29] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/29 18:58:31] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/29 18:58:33] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/29 18:58:34] ppgan.engine.trainer INFO: Metric psnr: 29.4280
    [05/29 18:58:34] ppgan.engine.trainer INFO: Metric ssim: 0.8011


## 四、模型评估
- 训练达到预期效果后，可以使用模型来测试。我已经提前将权重文件放在work/weight文件夹下，运行下行命令对测试集test_LR进行测试
- **注意**：训练过程中保存的权重是放在 PaddleGAN/output_dir/drn_psnr_x4_div2k-{time}/文件夹下，这里的time是和你开始训练的时间有关的，所以为了能让新手全流程跑通，不用修改路径的，就提前将权重拿出来放到weight文件夹下
- **如果想测试自己训练出的模型，更改--load 后面的模型权重路径即可**
- 测试的结果会在运行结束后保存在 PaddleGAN/output_dir/drn_psnr_x4_div2k-{time}/visual_test 文件夹下，**测试的结果**：PSNR:29.4236 SSIM:0.8002


```python
!python -u tools/main.py --config-file ../work/drn_psnr_x4_rssr.yaml --evaluate-only --load /home/aistudio/PaddleGAN/output_dir/drn_psnr_x4_rssr-2023-05-29-17-24/iter_20000_weight.pdparams
```

    [05/30 20:16:23] ppgan INFO: Configs: 
    dataset:
      test:
        gt_folder: data/RSdata_for_SR/test_HR
        lq_folder: data/RSdata_for_SR/test_LR/x4
        name: SRDataset
        preprocess:
        - key: lq
          name: LoadImageFromFile
        - key: gt
          name: LoadImageFromFile
        - input_keys:
          - lq
          - gt
          name: Transforms
          pipeline:
          - keys:
            - image
            - image
            name: Transpose
          - keys:
            - image
            - image
            mean:
            - 0.0
            - 0.0
            - 0.0
            name: Normalize
            std:
            - 1.0
            - 1.0
            - 1.0
        scale: 4
      train:
        batch_size: 8
        gt_folder: data/RSdata_for_SR/trian_HR
        lq_folder: data/RSdata_for_SR/train_LR/x4
        name: SRDataset
        num_workers: 4
        preprocess:
        - key: lq
          name: LoadImageFromFile
        - key: gt
          name: LoadImageFromFile
        - input_keys:
          - lq
          - gt
          name: Transforms
          output_keys:
          - lq
          - lqx2
          - gt
          pipeline:
          - gt_patch_size: 192
            keys:
            - image
            - image
            name: SRPairedRandomCrop
            scale: 4
            scale_list: true
          - keys:
            - image
            - image
            - image
            name: PairedRandomHorizontalFlip
          - keys:
            - image
            - image
            - image
            name: PairedRandomVerticalFlip
          - keys:
            - image
            - image
            - image
            name: PairedRandomTransposeHW
          - keys:
            - image
            - image
            - image
            name: Transpose
          - keys:
            - image
            - image
            - image
            mean:
            - 0.0
            - 0.0
            - 0.0
            name: Normalize
            std:
            - 1.0
            - 1.0
            - 1.0
        scale: 4
    is_train: false
    log_config:
      interval: 10
      visiual_interval: 500
    lr_scheduler:
      eta_min: 1.0e-07
      learning_rate: 0.0001
      name: CosineAnnealingRestartLR
      periods:
      - 1000000
      restart_weights:
      - 1
    min_max: !!python/tuple
    - 0.0
    - 255.0
    model:
      generator:
        n_blocks: 30
        n_colors: 3
        n_feats: 16
        name: DRNGenerator
        negval: 0.2
        rgb_range: 255
        scale: !!python/tuple
        - 2
        - 4
      name: DRN
      pixel_criterion:
        name: L1Loss
    optimizer:
      optimD:
        beta1: 0.9
        beta2: 0.999
        name: Adam
        net_names:
        - dual_model_0
        - dual_model_1
        weight_decay: 0.0
      optimG:
        beta1: 0.9
        beta2: 0.999
        name: Adam
        net_names:
        - generator
        weight_decay: 0.0
    output_dir: output_dir/drn_psnr_x4_rssr-2023-05-30-20-16
    profiler_options: null
    snapshot_config:
      interval: 1000
    timestamp: -2023-05-30-20-16
    total_iters: 20000
    validate:
      interval: 1000
      metrics:
        psnr:
          crop_border: 4
          name: PSNR
          test_y_channel: true
        ssim:
          crop_border: 4
          name: SSIM
          test_y_channel: true
      save_img: true
    
    W0530 20:16:23.638520  1875 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
    W0530 20:16:23.643198  1875 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [05/30 20:16:28] ppgan.engine.trainer INFO: Loaded pretrained weight for net generator
    [05/30 20:16:28] ppgan.engine.trainer INFO: Loaded pretrained weight for net dual_model_0
    [05/30 20:16:28] ppgan.engine.trainer INFO: Loaded pretrained weight for net dual_model_1
    [05/30 20:16:28] ppgan.engine.trainer INFO: Test iter: [0/420]
    [05/30 20:16:30] ppgan.engine.trainer INFO: Test iter: [10/420]
    [05/30 20:16:31] ppgan.engine.trainer INFO: Test iter: [20/420]
    [05/30 20:16:33] ppgan.engine.trainer INFO: Test iter: [30/420]
    [05/30 20:16:35] ppgan.engine.trainer INFO: Test iter: [40/420]
    [05/30 20:16:36] ppgan.engine.trainer INFO: Test iter: [50/420]
    [05/30 20:16:38] ppgan.engine.trainer INFO: Test iter: [60/420]
    [05/30 20:16:40] ppgan.engine.trainer INFO: Test iter: [70/420]
    [05/30 20:16:41] ppgan.engine.trainer INFO: Test iter: [80/420]
    [05/30 20:16:43] ppgan.engine.trainer INFO: Test iter: [90/420]
    [05/30 20:16:45] ppgan.engine.trainer INFO: Test iter: [100/420]
    [05/30 20:16:46] ppgan.engine.trainer INFO: Test iter: [110/420]
    [05/30 20:16:48] ppgan.engine.trainer INFO: Test iter: [120/420]
    [05/30 20:16:50] ppgan.engine.trainer INFO: Test iter: [130/420]
    [05/30 20:16:51] ppgan.engine.trainer INFO: Test iter: [140/420]
    [05/30 20:16:53] ppgan.engine.trainer INFO: Test iter: [150/420]
    [05/30 20:16:55] ppgan.engine.trainer INFO: Test iter: [160/420]
    [05/30 20:16:56] ppgan.engine.trainer INFO: Test iter: [170/420]
    [05/30 20:16:58] ppgan.engine.trainer INFO: Test iter: [180/420]
    [05/30 20:17:00] ppgan.engine.trainer INFO: Test iter: [190/420]
    [05/30 20:17:01] ppgan.engine.trainer INFO: Test iter: [200/420]
    [05/30 20:17:03] ppgan.engine.trainer INFO: Test iter: [210/420]
    [05/30 20:17:05] ppgan.engine.trainer INFO: Test iter: [220/420]
    [05/30 20:17:07] ppgan.engine.trainer INFO: Test iter: [230/420]
    [05/30 20:17:08] ppgan.engine.trainer INFO: Test iter: [240/420]
    [05/30 20:17:10] ppgan.engine.trainer INFO: Test iter: [250/420]
    [05/30 20:17:12] ppgan.engine.trainer INFO: Test iter: [260/420]
    [05/30 20:17:13] ppgan.engine.trainer INFO: Test iter: [270/420]
    [05/30 20:17:15] ppgan.engine.trainer INFO: Test iter: [280/420]
    [05/30 20:17:17] ppgan.engine.trainer INFO: Test iter: [290/420]
    [05/30 20:17:18] ppgan.engine.trainer INFO: Test iter: [300/420]
    [05/30 20:17:20] ppgan.engine.trainer INFO: Test iter: [310/420]
    [05/30 20:17:22] ppgan.engine.trainer INFO: Test iter: [320/420]
    [05/30 20:17:23] ppgan.engine.trainer INFO: Test iter: [330/420]
    [05/30 20:17:25] ppgan.engine.trainer INFO: Test iter: [340/420]
    [05/30 20:17:27] ppgan.engine.trainer INFO: Test iter: [350/420]
    [05/30 20:17:28] ppgan.engine.trainer INFO: Test iter: [360/420]
    [05/30 20:17:30] ppgan.engine.trainer INFO: Test iter: [370/420]
    [05/30 20:17:32] ppgan.engine.trainer INFO: Test iter: [380/420]
    [05/30 20:17:33] ppgan.engine.trainer INFO: Test iter: [390/420]
    [05/30 20:17:35] ppgan.engine.trainer INFO: Test iter: [400/420]
    [05/30 20:17:37] ppgan.engine.trainer INFO: Test iter: [410/420]
    [05/30 20:17:38] ppgan.engine.trainer INFO: Metric psnr: 29.4280
    [05/30 20:17:38] ppgan.engine.trainer INFO: Metric ssim: 0.8011


## 五、快速体验
- 定义一个预测类，快速调用训练好的模型权重对指定文件夹的低分辨率影像做4倍的超分，然后使用matplotlib可视化
- 接下来的示例，是对work/example/inputs文件夹下的图像进行重建，inputs文件夹中的文件为测试集中的图像。
- 运行下列命令，从测试集中复制图像到work/example/inputs文件夹中


```python
%cd /home/aistudio/work/example
!unzip NV10-dataset.zip
```

    /home/aistudio/work/example
    Archive:  NV10-dataset.zip
      inflating: NV10-dataset/images/233.jpg  
      inflating: NV10-dataset/images/237.jpg  
      inflating: NV10-dataset/images/625.jpg  
      inflating: NV10-dataset/images/185.jpg  
      inflating: NV10-dataset/images/270.jpg  
      inflating: NV10-dataset/images/328.jpg  
      inflating: NV10-dataset/images/355.jpg  
      inflating: NV10-dataset/images/628.jpg  
      inflating: NV10-dataset/images/036.jpg  
      inflating: NV10-dataset/images/055.jpg  
      inflating: NV10-dataset/images/120.jpg  
      inflating: NV10-dataset/images/228.jpg  
      inflating: NV10-dataset/images/325.jpg  
      inflating: NV10-dataset/images/419.jpg  
      inflating: NV10-dataset/images/473.jpg  
      inflating: NV10-dataset/images/609.jpg  
      inflating: NV10-dataset/images/038.jpg  
      inflating: NV10-dataset/images/048.jpg  
      inflating: NV10-dataset/images/069.jpg  
      inflating: NV10-dataset/images/024.jpg  
      inflating: NV10-dataset/images/143.jpg  
      inflating: NV10-dataset/images/300.jpg  
      inflating: NV10-dataset/images/375.jpg  
      inflating: NV10-dataset/images/455.jpg  
      inflating: NV10-dataset/images/032.jpg  
      inflating: NV10-dataset/images/091.jpg  
      inflating: NV10-dataset/images/145.jpg  
      inflating: NV10-dataset/images/509.jpg  
      inflating: NV10-dataset/images/557.jpg  
      inflating: NV10-dataset/images/060.jpg  
      inflating: NV10-dataset/images/346.jpg  
      inflating: NV10-dataset/images/506.jpg  
      inflating: NV10-dataset/images/378.jpg  
      inflating: NV10-dataset/images/084.jpg  
      inflating: NV10-dataset/images/303.jpg  
      inflating: NV10-dataset/images/308.jpg  
      inflating: NV10-dataset/images/507.jpg  
      inflating: NV10-dataset/images/548.jpg  
      inflating: NV10-dataset/images/009.jpg  
      inflating: NV10-dataset/images/023.jpg  
      inflating: NV10-dataset/images/097.jpg  
      inflating: NV10-dataset/images/103.jpg  
      inflating: NV10-dataset/images/123.jpg  
      inflating: NV10-dataset/images/389.jpg  
      inflating: NV10-dataset/images/373.jpg  
      inflating: NV10-dataset/images/562.jpg  
      inflating: NV10-dataset/images/618.jpg  
      inflating: NV10-dataset/images/149.jpg  
      inflating: NV10-dataset/images/152.jpg  
      inflating: NV10-dataset/images/258.jpg  
      inflating: NV10-dataset/images/495.jpg  
      inflating: NV10-dataset/images/170.jpg  
      inflating: NV10-dataset/images/205.jpg  
      inflating: NV10-dataset/images/360.jpg  
      inflating: NV10-dataset/images/436.jpg  
      inflating: NV10-dataset/images/450.jpg  
      inflating: NV10-dataset/images/458.jpg  
      inflating: NV10-dataset/images/528.jpg  
      inflating: NV10-dataset/images/588.jpg  
      inflating: NV10-dataset/images/074.jpg  
      inflating: NV10-dataset/images/132.jpg  
      inflating: NV10-dataset/images/356.jpg  
      inflating: NV10-dataset/images/581.jpg  
      inflating: NV10-dataset/images/317.jpg  
      inflating: NV10-dataset/images/537.jpg  
      inflating: NV10-dataset/images/552.jpg  
      inflating: NV10-dataset/images/361.jpg  
      inflating: NV10-dataset/images/497.jpg  
      inflating: NV10-dataset/images/567.jpg  
      inflating: NV10-dataset/images/015.jpg  
      inflating: NV10-dataset/images/016.jpg  
      inflating: NV10-dataset/images/051.jpg  
      inflating: NV10-dataset/images/293.jpg  
      inflating: NV10-dataset/images/329.jpg  
      inflating: NV10-dataset/images/453.jpg  
      inflating: NV10-dataset/images/641.jpg  
      inflating: NV10-dataset/images/067.jpg  
      inflating: NV10-dataset/images/106.jpg  
      inflating: NV10-dataset/images/234.jpg  
      inflating: NV10-dataset/images/505.jpg  
      inflating: NV10-dataset/images/127.jpg  
      inflating: NV10-dataset/images/272.jpg  
      inflating: NV10-dataset/images/438.jpg  
      inflating: NV10-dataset/images/156.jpg  
      inflating: NV10-dataset/images/445.jpg  
      inflating: NV10-dataset/images/576.jpg  
      inflating: NV10-dataset/images/359.jpg  
      inflating: NV10-dataset/images/363.jpg  
      inflating: NV10-dataset/images/395.jpg  
      inflating: NV10-dataset/images/479.jpg  
      inflating: NV10-dataset/images/080.jpg  
      inflating: NV10-dataset/images/102.jpg  
      inflating: NV10-dataset/images/250.jpg  
      inflating: NV10-dataset/images/295.jpg  
      inflating: NV10-dataset/images/029.jpg  
      inflating: NV10-dataset/images/071.jpg  
      inflating: NV10-dataset/images/290.jpg  
      inflating: NV10-dataset/images/487.jpg  
      inflating: NV10-dataset/images/073.jpg  
      inflating: NV10-dataset/images/428.jpg  
      inflating: NV10-dataset/images/457.jpg  
      inflating: NV10-dataset/images/446.jpg  
      inflating: NV10-dataset/images/595.jpg  
      inflating: NV10-dataset/images/030.jpg  
      inflating: NV10-dataset/images/274.jpg  
      inflating: NV10-dataset/images/336.jpg  
      inflating: NV10-dataset/images/202.jpg  
      inflating: NV10-dataset/images/246.jpg  
      inflating: NV10-dataset/images/027.jpg  
      inflating: NV10-dataset/images/050.jpg  
      inflating: NV10-dataset/images/197.jpg  
      inflating: NV10-dataset/images/399.jpg  
      inflating: NV10-dataset/images/481.jpg  
      inflating: NV10-dataset/images/566.jpg  
      inflating: NV10-dataset/images/630.jpg  
      inflating: NV10-dataset/images/019.jpg  
      inflating: NV10-dataset/images/058.jpg  
      inflating: NV10-dataset/images/369.jpg  
      inflating: NV10-dataset/images/396.jpg  
      inflating: NV10-dataset/images/410.jpg  
      inflating: NV10-dataset/images/496.jpg  
      inflating: NV10-dataset/images/040.jpg  
      inflating: NV10-dataset/images/041.jpg  
      inflating: NV10-dataset/images/320.jpg  
      inflating: NV10-dataset/images/554.jpg  
      inflating: NV10-dataset/images/063.jpg  
      inflating: NV10-dataset/images/465.jpg  
      inflating: NV10-dataset/images/539.jpg  
      inflating: NV10-dataset/images/130.jpg  
      inflating: NV10-dataset/images/138.jpg  
      inflating: NV10-dataset/images/309.jpg  
      inflating: NV10-dataset/images/368.jpg  
      inflating: NV10-dataset/images/049.jpg  
      inflating: NV10-dataset/images/054.jpg  
      inflating: NV10-dataset/images/122.jpg  
      inflating: NV10-dataset/images/269.jpg  
      inflating: NV10-dataset/images/319.jpg  
      inflating: NV10-dataset/images/406.jpg  
      inflating: NV10-dataset/images/529.jpg  
      inflating: NV10-dataset/images/644.jpg  
      inflating: NV10-dataset/images/649.jpg  
      inflating: NV10-dataset/images/066.jpg  
      inflating: NV10-dataset/images/238.jpg  
      inflating: NV10-dataset/images/344.jpg  
      inflating: NV10-dataset/images/267.jpg  
      inflating: NV10-dataset/images/326.jpg  
      inflating: NV10-dataset/images/354.jpg  
      inflating: NV10-dataset/images/459.jpg  
      inflating: NV10-dataset/images/550.jpg  
      inflating: NV10-dataset/images/006.jpg  
      inflating: NV10-dataset/images/112.jpg  
      inflating: NV10-dataset/images/236.jpg  
      inflating: NV10-dataset/images/125.jpg  
      inflating: NV10-dataset/images/544.jpg  
      inflating: NV10-dataset/images/624.jpg  
      inflating: NV10-dataset/images/104.jpg  
      inflating: NV10-dataset/images/292.jpg  
      inflating: NV10-dataset/images/404.jpg  
      inflating: NV10-dataset/images/211.jpg  
      inflating: NV10-dataset/images/240.jpg  
      inflating: NV10-dataset/images/474.jpg  
      inflating: NV10-dataset/images/020.jpg  
      inflating: NV10-dataset/images/089.jpg  
      inflating: NV10-dataset/images/131.jpg  
      inflating: NV10-dataset/images/391.jpg  
      inflating: NV10-dataset/images/411.jpg  
      inflating: NV10-dataset/images/239.jpg  
      inflating: NV10-dataset/images/307.jpg  
      inflating: NV10-dataset/images/310.jpg  
      inflating: NV10-dataset/images/460.jpg  
      inflating: NV10-dataset/images/565.jpg  
      inflating: NV10-dataset/images/007.jpg  
      inflating: NV10-dataset/images/096.jpg  
      inflating: NV10-dataset/images/172.jpg  
      inflating: NV10-dataset/images/136.jpg  
      inflating: NV10-dataset/images/259.jpg  
      inflating: NV10-dataset/images/477.jpg  
      inflating: NV10-dataset/images/519.jpg  
      inflating: NV10-dataset/images/541.jpg  
      inflating: NV10-dataset/images/044.jpg  
      inflating: NV10-dataset/images/121.jpg  
      inflating: NV10-dataset/images/218.jpg  
      inflating: NV10-dataset/images/612.jpg  
      inflating: NV10-dataset/images/189.jpg  
      inflating: NV10-dataset/images/332.jpg  
      inflating: NV10-dataset/images/600.jpg  
      inflating: NV10-dataset/images/451.jpg  
      inflating: NV10-dataset/images/501.jpg  
      inflating: NV10-dataset/images/582.jpg  
      inflating: NV10-dataset/images/637.jpg  
      inflating: NV10-dataset/images/157.jpg  
      inflating: NV10-dataset/images/379.jpg  
      inflating: NV10-dataset/images/429.jpg  
      inflating: NV10-dataset/images/456.jpg  
      inflating: NV10-dataset/images/476.jpg  
      inflating: NV10-dataset/images/491.jpg  
      inflating: NV10-dataset/images/646.jpg  
      inflating: NV10-dataset/images/650.jpg  
      inflating: NV10-dataset/images/153.jpg  
      inflating: NV10-dataset/images/195.jpg  
      inflating: NV10-dataset/images/277.jpg  
      inflating: NV10-dataset/images/510.jpg  
      inflating: NV10-dataset/images/353.jpg  
      inflating: NV10-dataset/images/358.jpg  
      inflating: NV10-dataset/images/467.jpg  
      inflating: NV10-dataset/images/555.jpg  
      inflating: NV10-dataset/images/614.jpg  
      inflating: NV10-dataset/images/086.jpg  
      inflating: NV10-dataset/images/113.jpg  
      inflating: NV10-dataset/images/151.jpg  
      inflating: NV10-dataset/images/330.jpg  
      inflating: NV10-dataset/images/433.jpg  
      inflating: NV10-dataset/images/470.jpg  
      inflating: NV10-dataset/images/490.jpg  
      inflating: NV10-dataset/images/599.jpg  
      inflating: NV10-dataset/images/035.jpg  
      inflating: NV10-dataset/images/056.jpg  
      inflating: NV10-dataset/images/262.jpg  
      inflating: NV10-dataset/images/643.jpg  
      inflating: NV10-dataset/images/603.jpg  
      inflating: NV10-dataset/images/629.jpg  
      inflating: NV10-dataset/images/430.jpg  
      inflating: NV10-dataset/images/488.jpg  
      inflating: NV10-dataset/images/586.jpg  
      inflating: NV10-dataset/images/245.jpg  
      inflating: NV10-dataset/images/255.jpg  
      inflating: NV10-dataset/images/324.jpg  
      inflating: NV10-dataset/images/147.jpg  
      inflating: NV10-dataset/images/206.jpg  
      inflating: NV10-dataset/images/221.jpg  
      inflating: NV10-dataset/images/407.jpg  
      inflating: NV10-dataset/images/531.jpg  
      inflating: NV10-dataset/images/011.jpg  
      inflating: NV10-dataset/images/061.jpg  
      inflating: NV10-dataset/images/315.jpg  
      inflating: NV10-dataset/images/440.jpg  
      inflating: NV10-dataset/images/553.jpg  
      inflating: NV10-dataset/images/095.jpg  
      inflating: NV10-dataset/images/144.jpg  
      inflating: NV10-dataset/images/190.jpg  
      inflating: NV10-dataset/images/364.jpg  
      inflating: NV10-dataset/images/424.jpg  
      inflating: NV10-dataset/images/119.jpg  
      inflating: NV10-dataset/images/129.jpg  
      inflating: NV10-dataset/images/203.jpg  
      inflating: NV10-dataset/images/520.jpg  
      inflating: NV10-dataset/images/535.jpg  
      inflating: NV10-dataset/images/611.jpg  
      inflating: NV10-dataset/images/002.jpg  
      inflating: NV10-dataset/images/282.jpg  
      inflating: NV10-dataset/images/454.jpg  
      inflating: NV10-dataset/images/088.jpg  
      inflating: NV10-dataset/images/414.jpg  
      inflating: NV10-dataset/images/632.jpg  
      inflating: NV10-dataset/images/210.jpg  
      inflating: NV10-dataset/images/494.jpg  
      inflating: NV10-dataset/images/583.jpg  
      inflating: NV10-dataset/images/385.jpg  
      inflating: NV10-dataset/images/504.jpg  
      inflating: NV10-dataset/images/116.jpg  
      inflating: NV10-dataset/images/264.jpg  
      inflating: NV10-dataset/images/365.jpg  
      inflating: NV10-dataset/images/575.jpg  
      inflating: NV10-dataset/images/291.jpg  
      inflating: NV10-dataset/images/390.jpg  
      inflating: NV10-dataset/images/563.jpg  
      inflating: NV10-dataset/images/573.jpg  
      inflating: NV10-dataset/images/615.jpg  
      inflating: NV10-dataset/images/636.jpg  
      inflating: NV10-dataset/images/018.jpg  
      inflating: NV10-dataset/images/252.jpg  
      inflating: NV10-dataset/images/257.jpg  
      inflating: NV10-dataset/images/521.jpg  
      inflating: NV10-dataset/images/527.jpg  
      inflating: NV10-dataset/images/619.jpg  
      inflating: NV10-dataset/images/452.jpg  
      inflating: NV10-dataset/images/062.jpg  
      inflating: NV10-dataset/images/158.jpg  
      inflating: NV10-dataset/images/374.jpg  
      inflating: NV10-dataset/images/039.jpg  
      inflating: NV10-dataset/images/357.jpg  
      inflating: NV10-dataset/images/626.jpg  
      inflating: NV10-dataset/images/461.jpg  
      inflating: NV10-dataset/images/593.jpg  
      inflating: NV10-dataset/images/043.jpg  
      inflating: NV10-dataset/images/083.jpg  
      inflating: NV10-dataset/images/196.jpg  
      inflating: NV10-dataset/images/466.jpg  
      inflating: NV10-dataset/images/610.jpg  
      inflating: NV10-dataset/images/616.jpg  
      inflating: NV10-dataset/images/193.jpg  
      inflating: NV10-dataset/images/331.jpg  
      inflating: NV10-dataset/images/412.jpg  
      inflating: NV10-dataset/images/426.jpg  
      inflating: NV10-dataset/images/480.jpg  
      inflating: NV10-dataset/images/199.jpg  
      inflating: NV10-dataset/images/401.jpg  
      inflating: NV10-dataset/images/402.jpg  
      inflating: NV10-dataset/images/393.jpg  
      inflating: NV10-dataset/images/523.jpg  
      inflating: NV10-dataset/images/645.jpg  
      inflating: NV10-dataset/images/042.jpg  
      inflating: NV10-dataset/images/241.jpg  
      inflating: NV10-dataset/images/311.jpg  
      inflating: NV10-dataset/images/377.jpg  
      inflating: NV10-dataset/images/572.jpg  
      inflating: NV10-dataset/images/235.jpg  
      inflating: NV10-dataset/images/322.jpg  
      inflating: NV10-dataset/images/362.jpg  
      inflating: NV10-dataset/images/498.jpg  
      inflating: NV10-dataset/images/515.jpg  
      inflating: NV10-dataset/images/013.jpg  
      inflating: NV10-dataset/images/194.jpg  
      inflating: NV10-dataset/images/431.jpg  
      inflating: NV10-dataset/images/622.jpg  
      inflating: NV10-dataset/images/313.jpg  
      inflating: NV10-dataset/images/574.jpg  
      inflating: NV10-dataset/images/587.jpg  
      inflating: NV10-dataset/images/092.jpg  
      inflating: NV10-dataset/images/154.jpg  
      inflating: NV10-dataset/images/226.jpg  
      inflating: NV10-dataset/images/296.jpg  
      inflating: NV10-dataset/images/405.jpg  
      inflating: NV10-dataset/images/101.jpg  
      inflating: NV10-dataset/images/183.jpg  
      inflating: NV10-dataset/images/231.jpg  
      inflating: NV10-dataset/images/447.jpg  
      inflating: NV10-dataset/images/536.jpg  
      inflating: NV10-dataset/images/605.jpg  
      inflating: NV10-dataset/images/647.jpg  
      inflating: NV10-dataset/images/026.jpg  
      inflating: NV10-dataset/images/207.jpg  
      inflating: NV10-dataset/images/441.jpg  
      inflating: NV10-dataset/images/604.jpg  
      inflating: NV10-dataset/images/214.jpg  
      inflating: NV10-dataset/images/370.jpg  
      inflating: NV10-dataset/images/489.jpg  
      inflating: NV10-dataset/images/381.jpg  
      inflating: NV10-dataset/images/397.jpg  
      inflating: NV10-dataset/images/483.jpg  
      inflating: NV10-dataset/images/500.jpg  
      inflating: NV10-dataset/images/142.jpg  
      inflating: NV10-dataset/images/148.jpg  
      inflating: NV10-dataset/images/223.jpg  
      inflating: NV10-dataset/images/533.jpg  
      inflating: NV10-dataset/images/222.jpg  
      inflating: NV10-dataset/images/388.jpg  
      inflating: NV10-dataset/images/427.jpg  
      inflating: NV10-dataset/images/261.jpg  
      inflating: NV10-dataset/images/547.jpg  
      inflating: NV10-dataset/images/059.jpg  
      inflating: NV10-dataset/images/178.jpg  
      inflating: NV10-dataset/images/184.jpg  
      inflating: NV10-dataset/images/100.jpg  
      inflating: NV10-dataset/images/118.jpg  
      inflating: NV10-dataset/images/124.jpg  
      inflating: NV10-dataset/images/305.jpg  
      inflating: NV10-dataset/images/423.jpg  
      inflating: NV10-dataset/images/017.jpg  
      inflating: NV10-dataset/images/037.jpg  
      inflating: NV10-dataset/images/064.jpg  
      inflating: NV10-dataset/images/640.jpg  
      inflating: NV10-dataset/images/409.jpg  
      inflating: NV10-dataset/images/099.jpg  
      inflating: NV10-dataset/images/318.jpg  
      inflating: NV10-dataset/images/387.jpg  
      inflating: NV10-dataset/images/176.jpg  
      inflating: NV10-dataset/images/569.jpg  
      inflating: NV10-dataset/images/564.jpg  
      inflating: NV10-dataset/images/081.jpg  
      inflating: NV10-dataset/images/392.jpg  
      inflating: NV10-dataset/images/394.jpg  
      inflating: NV10-dataset/images/376.jpg  
      inflating: NV10-dataset/images/463.jpg  
      inflating: NV10-dataset/images/556.jpg  
      inflating: NV10-dataset/images/591.jpg  
      inflating: NV10-dataset/images/045.jpg  
      inflating: NV10-dataset/images/078.jpg  
      inflating: NV10-dataset/images/181.jpg  
      inflating: NV10-dataset/images/254.jpg  
      inflating: NV10-dataset/images/526.jpg  
      inflating: NV10-dataset/images/627.jpg  
      inflating: NV10-dataset/images/633.jpg  
      inflating: NV10-dataset/images/068.jpg  
      inflating: NV10-dataset/images/082.jpg  
      inflating: NV10-dataset/images/225.jpg  
      inflating: NV10-dataset/images/585.jpg  
      inflating: NV10-dataset/images/256.jpg  
      inflating: NV10-dataset/images/321.jpg  
      inflating: NV10-dataset/images/327.jpg  
      inflating: NV10-dataset/images/065.jpg  
      inflating: NV10-dataset/images/304.jpg  
      inflating: NV10-dataset/images/442.jpg  
      inflating: NV10-dataset/images/542.jpg  
      inflating: NV10-dataset/images/110.jpg  
      inflating: NV10-dataset/images/134.jpg  
      inflating: NV10-dataset/images/137.jpg  
      inflating: NV10-dataset/images/416.jpg  
      inflating: NV10-dataset/images/538.jpg  
      inflating: NV10-dataset/images/057.jpg  
      inflating: NV10-dataset/images/075.jpg  
      inflating: NV10-dataset/images/323.jpg  
      inflating: NV10-dataset/images/072.jpg  
      inflating: NV10-dataset/images/230.jpg  
      inflating: NV10-dataset/images/340.jpg  
      inflating: NV10-dataset/images/502.jpg  
      inflating: NV10-dataset/images/175.jpg  
      inflating: NV10-dataset/images/334.jpg  
      inflating: NV10-dataset/images/335.jpg  
      inflating: NV10-dataset/images/271.jpg  
      inflating: NV10-dataset/images/432.jpg  
      inflating: NV10-dataset/images/464.jpg  
      inflating: NV10-dataset/images/471.jpg  
      inflating: NV10-dataset/images/503.jpg  
      inflating: NV10-dataset/images/001.jpg  
      inflating: NV10-dataset/images/164.jpg  
      inflating: NV10-dataset/images/198.jpg  
      inflating: NV10-dataset/images/444.jpg  
      inflating: NV10-dataset/images/508.jpg  
      inflating: NV10-dataset/images/115.jpg  
      inflating: NV10-dataset/images/159.jpg  
      inflating: NV10-dataset/images/177.jpg  
      inflating: NV10-dataset/images/285.jpg  
      inflating: NV10-dataset/images/337.jpg  
      inflating: NV10-dataset/images/383.jpg  
      inflating: NV10-dataset/images/114.jpg  
      inflating: NV10-dataset/images/155.jpg  
      inflating: NV10-dataset/images/253.jpg  
      inflating: NV10-dataset/images/518.jpg  
      inflating: NV10-dataset/images/584.jpg  
      inflating: NV10-dataset/images/224.jpg  
      inflating: NV10-dataset/images/486.jpg  
      inflating: NV10-dataset/images/517.jpg  
      inflating: NV10-dataset/images/341.jpg  
      inflating: NV10-dataset/images/403.jpg  
      inflating: NV10-dataset/images/425.jpg  
      inflating: NV10-dataset/images/439.jpg  
      inflating: NV10-dataset/images/443.jpg  
      inflating: NV10-dataset/images/034.jpg  
      inflating: NV10-dataset/images/098.jpg  
      inflating: NV10-dataset/images/200.jpg  
      inflating: NV10-dataset/images/511.jpg  
      inflating: NV10-dataset/images/551.jpg  
      inflating: NV10-dataset/images/204.jpg  
      inflating: NV10-dataset/images/279.jpg  
      inflating: NV10-dataset/images/281.jpg  
      inflating: NV10-dataset/images/478.jpg  
      inflating: NV10-dataset/images/534.jpg  
      inflating: NV10-dataset/images/008.jpg  
      inflating: NV10-dataset/images/090.jpg  
      inflating: NV10-dataset/images/174.jpg  
      inflating: NV10-dataset/images/623.jpg  
      inflating: NV10-dataset/images/294.jpg  
      inflating: NV10-dataset/images/631.jpg  
      inflating: NV10-dataset/images/133.jpg  
      inflating: NV10-dataset/images/168.jpg  
      inflating: NV10-dataset/images/219.jpg  
      inflating: NV10-dataset/images/216.jpg  
      inflating: NV10-dataset/images/613.jpg  
      inflating: NV10-dataset/images/094.jpg  
      inflating: NV10-dataset/images/469.jpg  
      inflating: NV10-dataset/images/513.jpg  
      inflating: NV10-dataset/images/522.jpg  
      inflating: NV10-dataset/images/543.jpg  
      inflating: NV10-dataset/images/248.jpg  
      inflating: NV10-dataset/images/350.jpg  
      inflating: NV10-dataset/images/367.jpg  
      inflating: NV10-dataset/images/276.jpg  
      inflating: NV10-dataset/images/413.jpg  
      inflating: NV10-dataset/images/549.jpg  
      inflating: NV10-dataset/images/343.jpg  
      inflating: NV10-dataset/images/514.jpg  
      inflating: NV10-dataset/images/621.jpg  
      inflating: NV10-dataset/images/033.jpg  
      inflating: NV10-dataset/images/162.jpg  
      inflating: NV10-dataset/images/249.jpg  
      inflating: NV10-dataset/images/475.jpg  
      inflating: NV10-dataset/images/594.jpg  
      inflating: NV10-dataset/images/638.jpg  
      inflating: NV10-dataset/images/025.jpg  
      inflating: NV10-dataset/images/215.jpg  
      inflating: NV10-dataset/images/380.jpg  
      inflating: NV10-dataset/images/243.jpg  
      inflating: NV10-dataset/images/398.jpg  
      inflating: NV10-dataset/images/608.jpg  
      inflating: NV10-dataset/images/003.jpg  
      inflating: NV10-dataset/images/232.jpg  
      inflating: NV10-dataset/images/299.jpg  
      inflating: NV10-dataset/images/339.jpg  
      inflating: NV10-dataset/images/579.jpg  
      inflating: NV10-dataset/images/046.jpg  
      inflating: NV10-dataset/images/166.jpg  
      inflating: NV10-dataset/images/188.jpg  
      inflating: NV10-dataset/images/516.jpg  
      inflating: NV10-dataset/images/580.jpg  
      inflating: NV10-dataset/images/022.jpg  
      inflating: NV10-dataset/images/111.jpg  
      inflating: NV10-dataset/images/485.jpg  
      inflating: NV10-dataset/images/366.jpg  
      inflating: NV10-dataset/images/560.jpg  
      inflating: NV10-dataset/images/571.jpg  
      inflating: NV10-dataset/images/597.jpg  
      inflating: NV10-dataset/images/169.jpg  
      inflating: NV10-dataset/images/182.jpg  
      inflating: NV10-dataset/images/525.jpg  
      inflating: NV10-dataset/images/288.jpg  
      inflating: NV10-dataset/images/558.jpg  
      inflating: NV10-dataset/images/093.jpg  
      inflating: NV10-dataset/images/139.jpg  
      inflating: NV10-dataset/images/268.jpg  
      inflating: NV10-dataset/images/386.jpg  
      inflating: NV10-dataset/images/244.jpg  
      inflating: NV10-dataset/images/351.jpg  
      inflating: NV10-dataset/images/418.jpg  
      inflating: NV10-dataset/images/297.jpg  
      inflating: NV10-dataset/images/420.jpg  
      inflating: NV10-dataset/images/482.jpg  
      inflating: NV10-dataset/images/141.jpg  
      inflating: NV10-dataset/images/263.jpg  
      inflating: NV10-dataset/images/275.jpg  
      inflating: NV10-dataset/images/180.jpg  
      inflating: NV10-dataset/images/348.jpg  
      inflating: NV10-dataset/images/592.jpg  
      inflating: NV10-dataset/images/492.jpg  
      inflating: NV10-dataset/images/201.jpg  
      inflating: NV10-dataset/images/301.jpg  
      inflating: NV10-dataset/images/434.jpg  
      inflating: NV10-dataset/images/302.jpg  
      inflating: NV10-dataset/images/345.jpg  
      inflating: NV10-dataset/images/530.jpg  
      inflating: NV10-dataset/images/642.jpg  
      inflating: NV10-dataset/images/005.jpg  
      inflating: NV10-dataset/images/087.jpg  
      inflating: NV10-dataset/images/126.jpg  
      inflating: NV10-dataset/images/512.jpg  
      inflating: NV10-dataset/images/559.jpg  
      inflating: NV10-dataset/images/052.jpg  
      inflating: NV10-dataset/images/109.jpg  
      inflating: NV10-dataset/images/187.jpg  
      inflating: NV10-dataset/images/161.jpg  
      inflating: NV10-dataset/images/179.jpg  
      inflating: NV10-dataset/images/242.jpg  
      inflating: NV10-dataset/images/349.jpg  
      inflating: NV10-dataset/images/352.jpg  
      inflating: NV10-dataset/images/028.jpg  
      inflating: NV10-dataset/images/085.jpg  
      inflating: NV10-dataset/images/128.jpg  
      inflating: NV10-dataset/images/648.jpg  
      inflating: NV10-dataset/images/010.jpg  
      inflating: NV10-dataset/images/053.jpg  
      inflating: NV10-dataset/images/590.jpg  
      inflating: NV10-dataset/images/251.jpg  
      inflating: NV10-dataset/images/287.jpg  
      inflating: NV10-dataset/images/298.jpg  
      inflating: NV10-dataset/images/338.jpg  
      inflating: NV10-dataset/images/435.jpg  
      inflating: NV10-dataset/images/047.jpg  
      inflating: NV10-dataset/images/167.jpg  
      inflating: NV10-dataset/images/217.jpg  
      inflating: NV10-dataset/images/589.jpg  
      inflating: NV10-dataset/images/372.jpg  
      inflating: NV10-dataset/images/382.jpg  
      inflating: NV10-dataset/images/468.jpg  
      inflating: NV10-dataset/images/532.jpg  
      inflating: NV10-dataset/images/620.jpg  
      inflating: NV10-dataset/images/140.jpg  
      inflating: NV10-dataset/images/209.jpg  
      inflating: NV10-dataset/images/266.jpg  
      inflating: NV10-dataset/images/417.jpg  
      inflating: NV10-dataset/images/601.jpg  
      inflating: NV10-dataset/images/171.jpg  
      inflating: NV10-dataset/images/284.jpg  
      inflating: NV10-dataset/images/333.jpg  
      inflating: NV10-dataset/images/606.jpg  
      inflating: NV10-dataset/images/635.jpg  
      inflating: NV10-dataset/images/163.jpg  
      inflating: NV10-dataset/images/400.jpg  
      inflating: NV10-dataset/images/421.jpg  
      inflating: NV10-dataset/images/316.jpg  
      inflating: NV10-dataset/images/540.jpg  
      inflating: NV10-dataset/images/617.jpg  
      inflating: NV10-dataset/images/077.jpg  
      inflating: NV10-dataset/images/186.jpg  
      inflating: NV10-dataset/images/312.jpg  
      inflating: NV10-dataset/images/546.jpg  
      inflating: NV10-dataset/images/117.jpg  
      inflating: NV10-dataset/images/384.jpg  
      inflating: NV10-dataset/images/449.jpg  
      inflating: NV10-dataset/images/499.jpg  
      inflating: NV10-dataset/images/079.jpg  
      inflating: NV10-dataset/images/108.jpg  
      inflating: NV10-dataset/images/448.jpg  
      inflating: NV10-dataset/images/602.jpg  
      inflating: NV10-dataset/images/191.jpg  
      inflating: NV10-dataset/images/306.jpg  
      inflating: NV10-dataset/images/422.jpg  
      inflating: NV10-dataset/images/107.jpg  
      inflating: NV10-dataset/images/192.jpg  
      inflating: NV10-dataset/images/220.jpg  
      inflating: NV10-dataset/images/484.jpg  
      inflating: NV10-dataset/images/568.jpg  
      inflating: NV10-dataset/images/004.jpg  
      inflating: NV10-dataset/images/012.jpg  
      inflating: NV10-dataset/images/076.jpg  
      inflating: NV10-dataset/images/314.jpg  
      inflating: NV10-dataset/images/437.jpg  
      inflating: NV10-dataset/images/524.jpg  
      inflating: NV10-dataset/images/634.jpg  
      inflating: NV10-dataset/images/031.jpg  
      inflating: NV10-dataset/images/146.jpg  
      inflating: NV10-dataset/images/213.jpg  
      inflating: NV10-dataset/images/545.jpg  
      inflating: NV10-dataset/images/173.jpg  
      inflating: NV10-dataset/images/283.jpg  
      inflating: NV10-dataset/images/415.jpg  
      inflating: NV10-dataset/images/265.jpg  
      inflating: NV10-dataset/images/577.jpg  
      inflating: NV10-dataset/images/150.jpg  
      inflating: NV10-dataset/images/229.jpg  
      inflating: NV10-dataset/images/260.jpg  
      inflating: NV10-dataset/images/212.jpg  
      inflating: NV10-dataset/images/289.jpg  
      inflating: NV10-dataset/images/607.jpg  
      inflating: NV10-dataset/images/639.jpg  
      inflating: NV10-dataset/images/021.jpg  
      inflating: NV10-dataset/images/070.jpg  
      inflating: NV10-dataset/images/208.jpg  
      inflating: NV10-dataset/images/371.jpg  
      inflating: NV10-dataset/images/561.jpg  
      inflating: NV10-dataset/images/135.jpg  
      inflating: NV10-dataset/images/160.jpg  
      inflating: NV10-dataset/images/280.jpg  
      inflating: NV10-dataset/images/347.jpg  
      inflating: NV10-dataset/images/408.jpg  
      inflating: NV10-dataset/images/472.jpg  
      inflating: NV10-dataset/images/493.jpg  
      inflating: NV10-dataset/images/165.jpg  
      inflating: NV10-dataset/images/247.jpg  
      inflating: NV10-dataset/images/278.jpg  
      inflating: NV10-dataset/images/598.jpg  
      inflating: NV10-dataset/images/014.jpg  
      inflating: NV10-dataset/images/462.jpg  
      inflating: NV10-dataset/images/570.jpg  
      inflating: NV10-dataset/images/286.jpg  
      inflating: NV10-dataset/images/342.jpg  
      inflating: NV10-dataset/images/578.jpg  
      inflating: NV10-dataset/images/596.jpg  
      inflating: NV10-dataset/images/105.jpg  
      inflating: NV10-dataset/images/227.jpg  
      inflating: NV10-dataset/images/273.jpg  



```python
# 运行这步会报已经存在文件夹的错，因为已经存在文件夹所以可以不执行这步，也可以删掉重新执行
!mkdir work/example
!mkdir work/example/inputs

# 复制影像
!cp PaddleGAN/data/RSdata_for_SR/test_LR/x4/runway16.png work/example/inputs/
!cp PaddleGAN/data/RSdata_for_SR/test_LR/x4/harbor13.png work/example/inputs/
!cp PaddleGAN/data/RSdata_for_SR/test_LR/x4/intersection16.png work/example/inputs/
!cp PaddleGAN/data/RSdata_for_SR/test_LR/x4/parkinglot61.png work/example/inputs/
!cp PaddleGAN/data/RSdata_for_SR/test_LR/x4/overpass63.png work/example/inputs/
!cp PaddleGAN/data/RSdata_for_SR/test_LR/x4/tenniscourt19.png work/example/inputs/
```

    /home/aistudio



```python
# 定义使用DRN模型预测的类DRNPredictor，需要输入参数：
# output: 模型输出保存的文件夹
# weight_path: 模型权重文件所在的路径
%cd /home/aistudio/PaddleGAN/
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import paddle 
from ppgan.models.generators import DRNGenerator
from ppgan.apps.base_predictor import BasePredictor
from ppgan.utils.logger import get_logger

class DRNPredictor(BasePredictor):
    def __init__(self, output='../work/example/output', weight_path=None):
        self.input = input
        self.output = os.path.join(output, 'DRN') #定义超分的结果保存的路径，为output路径+模型名所在文件夹
        self.model = DRNGenerator((2, 4)) # 实例化模型
        state_dict = paddle.load(weight_path) #加载权重
        state_dict = state_dict['generator'] 
        self.model.load_dict(state_dict)
        self.model.eval()
    # 标准化
    def norm(self, img):
        img = np.array(img).transpose([2, 0, 1]).astype('float32') / 1.0
        return img.astype('float32')
    # 去标准化
    def denorm(self, img):
        img = img.transpose((1, 2, 0))
        return (img * 1).clip(0, 255).astype('uint8')

    # 对图片输入进行预测，输入可以是图像路径，也可以是cv2读取的矩阵，或者PIL读取的图像文件
    def run_image(self, img):
        if isinstance(img, str):
            ori_img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            ori_img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            ori_img = img

        img = self.norm(ori_img) #图像标准化
        x = paddle.to_tensor(img[np.newaxis, ...]) #转成tensor
        with paddle.no_grad():
            out = self.model(x)[2] # 执行预测，DRN模型会输出三个tensor，第一个是原始低分辨率影像，第二个是放大两倍，第三个才是我们所需要的最后的结果
            

        pred_img = self.denorm(out.numpy()[0]) #tensor转成numpy的array并去标准化
        pred_img = Image.fromarray(pred_img) # array转图像
        return pred_img

    #输入图像文件路径
    def run(self, input):
        # 如果输出路径不存在则新建一个
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        pred_img = self.run_image(input) #对输入的图片进行预测
        out_path = None
        if self.output:
            try:
                base_name = os.path.splitext(os.path.basename(input))[0]
            except:
                base_name = 'result'
            out_path = os.path.join(self.output, base_name + '.png') #保存路径
            pred_img.save(out_path) #保存输出图片
            logger = get_logger()
            logger.info('Image saved to {}'.format(out_path))

        return pred_img, out_path
```

    /home/aistudio/PaddleGAN



```python
# png转成jpg
import os
import shutil

# 待修改后缀的文件夹路径
folder_path = '/home/aistudio/work/example/inputs'

# 将PNG文件路径列表赋值给变量png_files
png_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

# 遍历png_files列表，将每个PNG文件修改后缀为JPG，并保存到同文件夹下
for file in png_files:
    newfile = os.path.splitext(file)[0] + '.jpg'
    shutil.move(file, newfile)
```

- 定义好预测的类之后，接下来实例化预测类并对文件夹下的图像进行预测


- 在预测的过程中，展示输入的低分辨率图像与预测的图像
- 预测的结果保存在指定的文件夹的DRN文件夹中


```python
import matplotlib.pyplot as plt
import os
import numpy as np
%matplotlib inline

# 输出预测结果的文件夹
output = r'../work/example/output' 
# 模型路径
weight_path = r"../work/weight/drn_x4_rssr.pdparams"
# 待输入的低分辨率影像位置
input_dir = r"/home/aistudio/work/example/inputs" 

paddle.device.set_device("gpu:0") # 若是cpu环境，则替换为 paddle.device.set_device("cpu")
predictor = DRNPredictor(output, weight_path) # 实例化

filenames = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
for filename in filenames:
    imgPath = os.path.join(input_dir, filename)   
    outImg, _ = predictor.run(imgPath) # 预测
    # 可视化
    image = Image.open(imgPath)
    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1), plt.title('Input')
    plt.imshow(image), plt.axis('off')
    plt.subplot(1,2,2), plt.title('Output')
    plt.imshow(outImg), plt.axis('off') 
    plt.show()
```

    [05/30 19:04:57] ppgan INFO: Image saved to ../work/example/output/DRN/airplane40.png



    
![png](output_21_1.png)
    


    [05/30 19:04:58] ppgan INFO: Image saved to ../work/example/output/DRN/freeway12.png



    
![png](output_21_3.png)
    


    [05/30 19:04:58] ppgan INFO: Image saved to ../work/example/output/DRN/intersection28.png



    
![png](output_21_5.png)
    


    [05/30 19:04:59] ppgan INFO: Image saved to ../work/example/output/DRN/intersection99.png



    
![png](output_21_7.png)
    


    [05/30 19:04:59] ppgan INFO: Image saved to ../work/example/output/DRN/airplane86.png



    
![png](output_21_9.png)
    


    [05/30 19:05:00] ppgan INFO: Image saved to ../work/example/output/DRN/intersection32.png



    
![png](output_21_11.png)
    


    [05/30 19:05:00] ppgan INFO: Image saved to ../work/example/output/DRN/freeway00.png



    
![png](output_21_13.png)
    


    [05/30 19:05:00] ppgan INFO: Image saved to ../work/example/output/DRN/airplane87.png



    
![png](output_21_15.png)
    


    [05/30 19:05:01] ppgan INFO: Image saved to ../work/example/output/DRN/mediumresidential19.png



    
![png](output_21_17.png)
    


    [05/30 19:05:01] ppgan INFO: Image saved to ../work/example/output/DRN/freeway76.png



    
![png](output_21_19.png)
    


    [05/30 19:05:02] ppgan INFO: Image saved to ../work/example/output/DRN/freeway64.png



    
![png](output_21_21.png)
    


    [05/30 19:05:02] ppgan INFO: Image saved to ../work/example/output/DRN/intersection40.png



    
![png](output_21_23.png)
    


    [05/30 19:05:02] ppgan INFO: Image saved to ../work/example/output/DRN/airplane17.png



    
![png](output_21_25.png)
    


    [05/30 19:05:03] ppgan INFO: Image saved to ../work/example/output/DRN/intersection10.png



    
![png](output_21_27.png)
    


    [05/30 19:05:03] ppgan INFO: Image saved to ../work/example/output/DRN/intersection50.png



    
![png](output_21_29.png)
    


    [05/30 19:05:04] ppgan INFO: Image saved to ../work/example/output/DRN/airplane75.png



    
![png](output_21_31.png)
    


    [05/30 19:05:04] ppgan INFO: Image saved to ../work/example/output/DRN/intersection76.png



    
![png](output_21_33.png)
    


    [05/30 19:05:05] ppgan INFO: Image saved to ../work/example/output/DRN/intersection87.png



    
![png](output_21_35.png)
    


    [05/30 19:05:05] ppgan INFO: Image saved to ../work/example/output/DRN/intersection26.png



    
![png](output_21_37.png)
    


    [05/30 19:05:06] ppgan INFO: Image saved to ../work/example/output/DRN/intersection89.png



    
![png](output_21_39.png)
    


    [05/30 19:05:06] ppgan INFO: Image saved to ../work/example/output/DRN/freeway78.png



    
![png](output_21_41.png)
    


    [05/30 19:05:06] ppgan INFO: Image saved to ../work/example/output/DRN/airplane65.png



    
![png](output_21_43.png)
    


    [05/30 19:05:07] ppgan INFO: Image saved to ../work/example/output/DRN/airplane48.png



    
![png](output_21_45.png)
    


    [05/30 19:05:07] ppgan INFO: Image saved to ../work/example/output/DRN/freeway07.png



    
![png](output_21_47.png)
    


    [05/30 19:05:08] ppgan INFO: Image saved to ../work/example/output/DRN/airplane09.png



    
![png](output_21_49.png)
    


    [05/30 19:05:08] ppgan INFO: Image saved to ../work/example/output/DRN/freeway75.png



    
![png](output_21_51.png)
    


    [05/30 19:05:09] ppgan INFO: Image saved to ../work/example/output/DRN/freeway93.png



    
![png](output_21_53.png)
    


    [05/30 19:05:09] ppgan INFO: Image saved to ../work/example/output/DRN/intersection19.png



    
![png](output_21_55.png)
    


    [05/30 19:05:10] ppgan INFO: Image saved to ../work/example/output/DRN/airplane32.png



    
![png](output_21_57.png)
    


    [05/30 19:05:10] ppgan INFO: Image saved to ../work/example/output/DRN/freeway44.png



    
![png](output_21_59.png)
    


    [05/30 19:05:11] ppgan INFO: Image saved to ../work/example/output/DRN/intersection30.png



    
![png](output_21_61.png)
    


    [05/30 19:05:11] ppgan INFO: Image saved to ../work/example/output/DRN/mediumresidential15.png



    
![png](output_21_63.png)
    


    [05/30 19:05:11] ppgan INFO: Image saved to ../work/example/output/DRN/airplane55.png



    
![png](output_21_65.png)
    


    [05/30 19:05:12] ppgan INFO: Image saved to ../work/example/output/DRN/airplane71.png



    
![png](output_21_67.png)
    


    [05/30 19:05:12] ppgan INFO: Image saved to ../work/example/output/DRN/airplane04.png



    
![png](output_21_69.png)
    


    [05/30 19:05:13] ppgan INFO: Image saved to ../work/example/output/DRN/airplane28.png



    
![png](output_21_71.png)
    


    [05/30 19:05:13] ppgan INFO: Image saved to ../work/example/output/DRN/freeway67.png



    
![png](output_21_73.png)
    


    [05/30 19:05:13] ppgan INFO: Image saved to ../work/example/output/DRN/airplane25.png



    
![png](output_21_75.png)
    


    [05/30 19:05:14] ppgan INFO: Image saved to ../work/example/output/DRN/intersection77.png



    
![png](output_21_77.png)
    


    [05/30 19:05:14] ppgan INFO: Image saved to ../work/example/output/DRN/freeway55.png



    
![png](output_21_79.png)
    


    [05/30 19:05:15] ppgan INFO: Image saved to ../work/example/output/DRN/airplane80.png



    
![png](output_21_81.png)
    


    [05/30 19:05:15] ppgan INFO: Image saved to ../work/example/output/DRN/freeway03.png



    
![png](output_21_83.png)
    


    [05/30 19:05:15] ppgan INFO: Image saved to ../work/example/output/DRN/airplane85.png



    
![png](output_21_85.png)
    


    [05/30 19:05:16] ppgan INFO: Image saved to ../work/example/output/DRN/freeway26.png



    
![png](output_21_87.png)
    


    [05/30 19:05:16] ppgan INFO: Image saved to ../work/example/output/DRN/intersection70.png



    
![png](output_21_89.png)
    


    [05/30 19:05:17] ppgan INFO: Image saved to ../work/example/output/DRN/airplane81.png



    
![png](output_21_91.png)
    


    [05/30 19:05:17] ppgan INFO: Image saved to ../work/example/output/DRN/airplane74.png



    
![png](output_21_93.png)
    


    [05/30 19:05:17] ppgan INFO: Image saved to ../work/example/output/DRN/freeway25.png



    
![png](output_21_95.png)
    


    [05/30 19:05:18] ppgan INFO: Image saved to ../work/example/output/DRN/intersection16.png



    
![png](output_21_97.png)
    


    [05/30 19:05:18] ppgan INFO: Image saved to ../work/example/output/DRN/airplane30.png



    
![png](output_21_99.png)
    


    [05/30 19:05:18] ppgan INFO: Image saved to ../work/example/output/DRN/freeway77.png



    
![png](output_21_101.png)
    


    [05/30 19:05:19] ppgan INFO: Image saved to ../work/example/output/DRN/intersection43.png


## 六、总结
- 使用PaddleGAN进行迁移学习对遥感影像进行超分辨率，只需要2个小时即可达到上述效果，对于算力不够的小伙伴可以尝试。
- DRN网络的论文原文实验结果上来看，效果比RCAN略高，有兴趣做对比的，可以结合[以RCAN模型对遥感图像超分辨率重建，可以直接体验！](https://aistudio.baidu.com/aistudio/projectdetail/3508912)项目做一个对比，视觉效果上看是差不多的。


```python

```
