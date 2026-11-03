# 来源

- π0.5模型、JAX训练器、LoRA变体、归一化、图像变换和tokenizer来自Physical Intelligence OpenPI，固定commit`215abfb217dbac7d5f1273282331b9b1866c0479`；源码、Apache-2.0与Gemma相关许可在服务器独立副本`vendor/openpi`保留。
- 基础模型为官方`gs://openpi-assets/checkpoints/pi05_base/params`，不加载已训练的ACT或黄块任务模型。
- 折毛巾转换数据来自用户提供51段原始遥操作录制；之前ACT的`FuFutheBest/yam_train`转换结果只读复用，并校验文件SHA。数据中的物理标签源和camera映射保留。
- 新增的YAM折毛巾数据读取、输入输出适配、分组比较、CPU预检、独立配置与实验工具属于本次实现；12关节delta／夹爪绝对约定与OpenPI官方ALOHA接口的mask一致，但不复用ALOHA硬件符号或夹爪换算。
- 旧π0.5 YAM适配器仅用于核对其接口差异；本次不复制其未来从臂标签，也不把它已有任务结果当作折毛巾结果。

- 独立操作包复用已保留的FuFutheBest/yam_infer固定版本（bb2e8aba33d6bd9bcba43c2ffca129ad90c3feb5）运行器、相机、录制、i2rt控制与许可证；仅在新副本替换模型桥接、任务配置、计划截止时间和故障退出处理。原ACT目录未修改。
