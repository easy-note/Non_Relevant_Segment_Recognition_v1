# Vi-HUB Pro (Out of Body Recognition Project)
## Model Zoo
- MobileNet-V3-large
  - https://drive.google.com/drive/u/0/folders/1tEMeVJldnULGG7K1Q5u2f7zeKFXzk1no
- EfficientNet-B3
  - https://drive.google.com/drive/u/0/folders/1E9iETK3i-2XBZIEYW5x7FWIDvM9rCL_n
---
## Model summary
MobileNet-V3-large (Pre-trained)
- MODEL PARAMS
  ```
  ======================================================================
  Layer (type:depth-idx)                   Output Shape              Param #
  ======================================================================
  ├─MobileNetV3: 1-1                       [-1, 2]                   --
  |    └─Sequential: 2-1                   [-1, 960, 7, 7]           --
  |    |    └─ConvBNActivation: 3-1        [-1, 16, 112, 112]        464
  |    |    └─InvertedResidual: 3-2        [-1, 16, 112, 112]        464
  |    |    └─InvertedResidual: 3-3        [-1, 24, 56, 56]          3,440
  |    |    └─InvertedResidual: 3-4        [-1, 24, 56, 56]          4,440
  |    |    └─InvertedResidual: 3-5        [-1, 40, 28, 28]          10,328
  |    |    └─InvertedResidual: 3-6        [-1, 40, 28, 28]          20,992
  |    |    └─InvertedResidual: 3-7        [-1, 40, 28, 28]          20,992
  |    |    └─InvertedResidual: 3-8        [-1, 80, 14, 14]          32,080
  |    |    └─InvertedResidual: 3-9        [-1, 80, 14, 14]          34,760
  |    |    └─InvertedResidual: 3-10       [-1, 80, 14, 14]          31,992
  |    |    └─InvertedResidual: 3-11       [-1, 80, 14, 14]          31,992
  |    |    └─InvertedResidual: 3-12       [-1, 112, 14, 14]         214,424
  |    |    └─InvertedResidual: 3-13       [-1, 112, 14, 14]         386,120
  |    |    └─InvertedResidual: 3-14       [-1, 160, 7, 7]           429,224
  |    |    └─InvertedResidual: 3-15       [-1, 160, 7, 7]           797,360
  |    |    └─InvertedResidual: 3-16       [-1, 160, 7, 7]           797,360
  |    |    └─ConvBNActivation: 3-17       [-1, 960, 7, 7]           155,520
  |    └─AdaptiveAvgPool2d: 2-2            [-1, 960, 1, 1]           --
  |    └─Sequential: 2-3                   [-1, 2]                   --
  |    |    └─Linear: 3-18                 [-1, 1280]                1,230,080
  |    |    └─Hardswish: 3-19              [-1, 1280]                --
  |    |    └─Dropout: 3-20                [-1, 1280]                --
  |    |    └─Linear: 3-21                 [-1, 2]                   2,562
  ======================================================================
  Total params: 4,204,594
  Trainable params: 4,204,594
  Non-trainable params: 0
  Total mult-adds (M): 28.30
  ======================================================================
  Input size (MB): 0.57
  Forward/backward pass size (MB): 3.79
  Params size (MB): 16.04
  Estimated Total Size (MB): 20.40
  ======================================================================
  ```
- DATASET (ROBOT, FOLD 1)
  - Train dataset [80]
    ```
    'R_1', 'R_3', 'R_4', 'R_5', 'R_7', 'R_10', 'R_14', 'R_15', 'R_17', 'R_18',
    'R_19', 'R_22', 'R_48', 'R_56', 'R_76', 'R_84', 'R_94', 'R_116', 'R_117', 'R_201',
    'R_203', 'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_303', 'R_304',
    'R_305', 'R_310', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_338', 'R_339', 'R_340',
    'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357', 'R_358', 'R_369',
    'R_372', 'R_376', 'R_378', 'R_379', 'R_391', 'R_393', 'R_399', 'R_400', 'R_402', 'R_403',
    'R_406', 'R_409', 'R_412', 'R_413', 'R_415', 'R_419', 'R_420', 'R_427', 'R_436', 'R_445',
    'R_449', 'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_532', 'R_533'
    ```
  - Validation dataset [20]
    ```
    'R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312',
    'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526'
    ```
- MODEL TUNING
  - Optimizer `Adam`
  - Criterion `Cross Entropy Loss`
  - Scheduler `StepLR (step size=10, gamma=0.1)`
  - Epoch `min:25, max:50`
  - Batch size `32`
  - Init lr `0.001`
  - Ramdom seed `10`
  - Train IB:OOB ratio `3:1`
EfficientNet-B3 (Pre-trained)
- MODEL PARAMS
  ```
  ======================================================================
  Layer (type:depth-idx)                                  Output Shape              Param #
  ======================================================================
  ├─EfficientNet: 1-1                                     [-1, 2]                   --
  |    └─Conv2dStaticSamePadding: 2-1                     [-1, 40, 112, 112]        --
  |    |    └─ZeroPad2d: 3-1                              [-1, 3, 225, 225]         --
  |    └─BatchNorm2d: 2-2                                 [-1, 40, 112, 112]        80
  |    └─MemoryEfficientSwish: 2-3                        [-1, 40, 112, 112]        --
  |    └─ModuleList: 2                                    []                        --
  |    |    └─MBConvBlock: 3-2                            [-1, 24, 112, 112]        2,298
  |    |    └─MBConvBlock: 3-3                            [-1, 24, 112, 112]        1,206
  |    |    └─MBConvBlock: 3-4                            [-1, 32, 56, 56]          11,878
  |    |    └─MBConvBlock: 3-5                            [-1, 32, 56, 56]          18,120
  |    |    └─MBConvBlock: 3-6                            [-1, 32, 56, 56]          18,120
  |    |    └─MBConvBlock: 3-7                            [-1, 48, 28, 28]          24,296
  |    |    └─MBConvBlock: 3-8                            [-1, 48, 28, 28]          43,308
  |    |    └─MBConvBlock: 3-9                            [-1, 48, 28, 28]          43,308
  |    |    └─MBConvBlock: 3-10                           [-1, 96, 14, 14]          52,620
  |    |    └─MBConvBlock: 3-11                           [-1, 96, 14, 14]          146,520
  |    |    └─MBConvBlock: 3-12                           [-1, 96, 14, 14]          146,520
  |    |    └─MBConvBlock: 3-13                           [-1, 96, 14, 14]          146,520
  |    |    └─MBConvBlock: 3-14                           [-1, 96, 14, 14]          146,520
  |    |    └─MBConvBlock: 3-15                           [-1, 136, 14, 14]         178,856
  |    |    └─MBConvBlock: 3-16                           [-1, 136, 14, 14]         302,226
  |    |    └─MBConvBlock: 3-17                           [-1, 136, 14, 14]         302,226
  |    |    └─MBConvBlock: 3-18                           [-1, 136, 14, 14]         302,226
  |    |    └─MBConvBlock: 3-19                           [-1, 136, 14, 14]         302,226
  |    |    └─MBConvBlock: 3-20                           [-1, 232, 7, 7]           380,754
  |    |    └─MBConvBlock: 3-21                           [-1, 232, 7, 7]           849,642
  |    |    └─MBConvBlock: 3-22                           [-1, 232, 7, 7]           849,642
  |    |    └─MBConvBlock: 3-23                           [-1, 232, 7, 7]           849,642
  |    |    └─MBConvBlock: 3-24                           [-1, 232, 7, 7]           849,642
  |    |    └─MBConvBlock: 3-25                           [-1, 232, 7, 7]           849,642
  |    |    └─MBConvBlock: 3-26                           [-1, 384, 7, 7]           1,039,258
  |    |    └─MBConvBlock: 3-27                           [-1, 384, 7, 7]           2,244,960
  |    └─Conv2dStaticSamePadding: 2-4                     [-1, 1536, 7, 7]          --
  |    |    └─Identity: 3-28                              [-1, 384, 7, 7]           --
  |    └─BatchNorm2d: 2-5                                 [-1, 1536, 7, 7]          3,072
  |    └─MemoryEfficientSwish: 2-6                        [-1, 1536, 7, 7]          --
  |    └─AdaptiveAvgPool2d: 2-7                           [-1, 1536, 1, 1]          --
  |    └─Dropout: 2-8                                     [-1, 1536]                --
  |    └─Linear: 2-9                                      [-1, 2]                   3,074
  ======================================================================
  Total params: 10,108,402
  Trainable params: 10,108,402
  Non-trainable params: 0
  Total mult-adds (M): 981.89
  ======================================================================
  Input size (MB): 0.57
  Forward/backward pass size (MB): 99.08
  Params size (MB): 38.56
  Estimated Total Size (MB): 138.21
  ======================================================================
  ```
- DATASET (ROBOT, FOLD 1)
  - Train dataset [80]
    ```
    'R_1', 'R_3', 'R_4', 'R_5', 'R_7', 'R_10', 'R_14', 'R_15', 'R_17', 'R_18',
    'R_19', 'R_22', 'R_48', 'R_56', 'R_76', 'R_84', 'R_94', 'R_116', 'R_117', 'R_201',
    'R_203', 'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_303', 'R_304',
    'R_305', 'R_310', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_338', 'R_339', 'R_340',
    'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357', 'R_358', 'R_369',
    'R_372', 'R_376', 'R_378', 'R_379', 'R_391', 'R_393', 'R_399', 'R_400', 'R_402', 'R_403',
    'R_406', 'R_409', 'R_412', 'R_413', 'R_415', 'R_419', 'R_420', 'R_427', 'R_436', 'R_445',
    'R_449', 'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_532', 'R_533'
    ```
  - Validation dataset [20]
    ```
    'R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312',
    'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526'
    ```
- MODEL TUNING
  - Optimizer `Adam`
  - Criterion `Cross Entropy Loss`
  - Scheduler `StepLR (step size=10, gamma=0.1)`
  - Epoch `min:25, max:50`
  - Batch size `32`
  - Init lr `0.001`
  - Ramdom seed `10`
  - Train IB:OOB ratio `3:1`
---
## Model Test
Patient index [5]
```
R_301, R_312, R_405, R_418, R_526
```
- 각 환자별 CR, OR result (validation result) :
  - https://docs.google.com/presentation/d/1_oSgGzAvWH-o2MFhJ0RpPPGqg2f2yG9Cd8CVmpmVK10/edit#slide=id.ge035b7fe46_0_1080

- Patient video path
  - NAS2 (192.168.16.106)
    - ID : hutom_access
    - PW : Hutom0615!
```
R_301_ch1_01: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_301_ch1_01.mp4
R_301_ch1_04: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_301_ch1_04.mp4
R_312_ch1_02: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_312_ch1_02.mp4
R_312_ch1_03: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_312_ch1_03.mp4
R_405_ch1_01: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_405_ch1_01.mp4
R_405_ch1_03: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_405_ch1_03.mp4
R_405_ch1_05: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_405_ch1_05.mp4
R_418_ch1_02: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_418_ch1_02.mp4
R_418_ch1_04: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_418_ch1_04.mp4
R_418_ch1_06: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_418_ch1_06.mp4
R_526_ch1_01: /hSDB_all/VIDEO/Gastrectomy/Robot/01_G_01_R_526_ch1_01.mp4
```
