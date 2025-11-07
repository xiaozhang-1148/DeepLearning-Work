# 项目说明

## DataSet

### 数据集说明
- 数据集总量：`9911份`
- 1、组员手工切分 并 标注 `1000份`;
- 2、使用公共数据集`8911份`;

## 实验说明 —— 训练设备为 1 张 英伟达A10 GPU ，进行 7 轮实验

### 实验内容
#### 前 4 轮使用 AdamW 优化器进行训练

- version_1 : 以 学习率`3e-4`，不开启`数据增强`,未启用`动态学习率`,进行`Epoch:100`训练 → 作为实验 baseline;
```
实验结果 :
val_ExpRate : 53.83%
test:
Correct (ExpRate): 51.87%
Correct <= 1 error: 69.32%
Correct <= 2 errors: 78.30%
Correct <= 3 errors: 82.54%
```

### 未开启数据增强数据集
<img src = "实验_logger_image/未开启数据增强_samples.png">

- version_2: 以 学习率`2e-3`, 开启`数据增强`, 启用`动态学习率` → 识别准确率显著提升，实现该数据集下的最佳性能;
```
实验结果 :
val_ExpRate : 57.86%
test:
Correct (ExpRate): 53.48%
Correct <= 1 error: 70.33%
Correct <= 2 errors: 79.21%
Correct <= 3 errors: 82.74%
```

### 开启数据增强数据集
<img src = "实验_logger_image/开启数据增强_samples.png">

- version_3: 增加 decoder 层数以提升模型容量 → [ num_decoder_layers : 3 → 4 ]
```
实验结果 : 模型层次太深，loss下降困难，无法收敛
```

<img src = "实验_logger_image/增加模型_train_loss.png">

- version_4: 提升 drouput 0.3 → 0.5 → 【 正则化过高，会降低 ExpRate，适当减小 dropout 】
```
实验结果 : 
val_ExpRate : 53.63%
test:
Correct (ExpRate): 51.16%
Correct <= 1 error: 68.52%
Correct <= 2 errors: 77.40%
Correct <= 3 errors: 81.74%
```

### AdamW 优化器学习率变化曲线
<img src = "实验_logger_image/Adaw_learning_rate.png">

### AdamW_train_loss
<img src = "实验_logger_image/AdamW_train_loss_image.png">

### AdamW_val_ExpRate
<img src = "实验_logger_image/AdamW_val_Exprate_image.png">

#### 后 3 轮使用 SGD 优化器，进行对比实验
- version_5 : 更换优化器 → `SGD` , 学习率调整为 0.08(论文使用学习率);

```
此处观察到，在完成 `Epoch:100` 训练之后，train_loss并未收敛，因此 `Epoch:100 → 200`
```
### SGD_train_loss
<img src = "实验_logger_image/SGD_train_loss_image.png">

- version_6: 进行`Epoch:200`训练;
```
val_Exprate : 52.12%
test:
Correct (ExpRate): 47.43%
Correct <= 1 error: 66.90%
Correct <= 2 errors: 76.19%
Correct <= 3 errors: 80.02%
```

- version_7: 进行`Epoch:300`训练;
```
val_Exprate : 54.33%
test:
Correct (ExpRate): 50.45%
Correct <= 1 error: 68.62%
Correct <= 2 errors: 77.60%
Correct <= 3 errors: 82.14%
```
<img src = "实验_logger_image/SGD_learning_rate.png">


## 最终实验结果
<img src = "实验_logger_image/val_ExpRate_img.png" alt="val_ExpRate">
<img src = "实验_logger_image/train_Loss_image.png">


# 使用大样本进行训练，探索模型的最佳性能

## 使用大样本数据（HME100K）—— 80K数据进行训练，10K数据作为验证集

- config : Epoch:`200` ; learning_Rate: `3e-2` ; `开启数据增强`

### train_loss
<img src = "实验_logger_image/大样本数据集_train_loss.png" alt="train_loss">

### val_ExpRate
<img src = "实验_logger_image/大样本数据集_val_ExpRate.png" alt="val_ExpRate">

```
使用 相同数据集进行测试
val_ExpRate = 68.85%
test:
Correct (ExpRate): 80.52%
Correct <= 1 error: 90.41%
Correct <= 2 errors: 93.84%
Correct <= 3 errors: 95.26%
```

## 总结：使用大样本训练能够实现更优越的性能，且在原数据集进行验证时，能够实现更优的ExpRate

### ExpRate 验证指标说明
```
    - 计算公式 ： ExpRate = （latex完全匹配的数量） / （测试集所有表达式数量）
    - Correct <= 1 error = （latex完全匹配的数量 + 出现一个字符错误的latex表达式） / （测试集所有表达式数量）
    - 以此类推
```