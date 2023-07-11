# Welcom to the Catty
任何一个训练脚本，应该被拆分成dataloader，criterion，models，engine，然后就能享受统一的dataloader框架、transform、tricks、logs，而非重复地在不同的训练脚本中写相似的内容。

## TODO
- 框架结构组织不够清晰、缺乏框架图
- 如何更轻易地拆解现存的训练代码
- 如何删减部分冗余内容，现在改起来特别吃力

## 文件说明
### configs
> 参照 [Swin Transformer](https://github.com/microsoft/Swin-Transformer) 代码，默认配置文件config.py也来自于此处
- 该目录下以项目划分，注意：每一个项目都应该有一个defualt配置文件，以免在不同配置文件下缺失配置的情况
- 可以直接在yaml文件中添加新的config项，而无需提前在config.py中添加
- TODO: [yacs](https://github.com/rbgirshick/yacs)库对类型的检测过于严格，能否进行小改？
### dataloader
> 以PyTorch和timm为基础并做了一些兼容性的工作，有想法融入[DALI](https://github.com/NVIDIA/DALI)，但个人体验下来麻烦程度大于效率增益
- 支持完全复制现存的transform和dataset
- 完全支持timm的config写法，以获得统一的transform和dataloader
- TODO: `K-Fold Validation`的实现有问题
- TODO: `Multiple-Transform`的抽象做的很差
### engine
> 个人的想象大于借鉴，可能需要参考[OpenMMLab](https://github.com/open-mmlab)
- `base.py`里的基类主要承载了训练tricks和log，将具体的复现方法的代码抽离到了`engine`中
- `engine`主要由不同`hooks`构成并持久化每个方法的自定义变量。*这里想模仿各大成熟框架，但能力太差，这部分不够清晰*
- TODO: 构建清晰的结构组织图，这里较为混乱
### models
> 几乎等于timm，将大部分需要复现的方法模型统一了一下
- 记得一定要注册模型，不然没办法创建，整个框架的这部分都遵循了timm的架构
- TODO：如何加强复现方法模型代码的统一性？
### optimizer & criterion & scheduler
> 几乎等于timm，做了一些兼容性的工作
## Tips
- Please change the timm default transform code, test_center_crop，默认这里是用的短边，然后保持比例的scale，如果想不保持比例就要对代码进行修改，不然就不行
- shell中添加bool要用`True`，yaml文件中用`true`