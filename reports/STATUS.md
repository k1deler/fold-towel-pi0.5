# 当前状态：π0.5微调准备完成，未开始训练

- 本地终端可运行；训练服务器SSH已认证并完成实际只读检查，未输出或上传认证副本。
- ACT保留：51段、28,271帧；已完成5000epoch和25,000更新；选出epoch3499，旧模型SHA以`provenance.json`记录为准，没有改动。
- 新π0.5目录：`~/output/personal/k1deler/fold_towel_pi05_20261102`；代码与OpenPI固定版本副本分离，既有共享源码／环境未修改。
- 服务器已有约12.44GB的`pi05_base/params`及OpenPI Python3.11／JAX0.5.3／Flax0.10.2环境。只检查了基础权重元数据，不等于整个基础模型已成功恢复和反向更新。
- 24项CPU测试通过；对真实数据进行了SHA复核、相同ACT episode划分检查、训练集分位数拟合。
- 40段训练／11段验证；使用50步完整目标窗口，训练20,070个、验证5,702个。窗口不跨episode，不以零填补尾部标签。
- 原OpenPI真实DataLoader批次`[4,50,32]`通过，模型图像224×224、prompt/state tokens200、缺失base图像false掩码通过；物理动作归一化／还原往返最大误差0。
- π0.5配置为12关节delta、夹爪absolute；ACT原合同是14维delta。两者的物理leader目标一致，表示不同，不能直接套用ACT输出还原。
- 代码与统计的SHA凭证已记录，修改后须重新preflight；正式train必须先完成两步真实GPU LoRA烟雾训练并生成新checkpoint。

**尚未完成**：π0.5基础参数实际恢复、GPU loss／gradient检查、真实微调、微调后验证集指标、机器人部署和实机折叠。当前报告不能当作这些步骤已通过。

本项目没有机器人启动入口。所有命令仅面向文件、CPU或显式选择的GPU。新仓库为`k1deler/fold-towel-`，不要与ACT仓库`k1deler/fold-towel`混淆。
