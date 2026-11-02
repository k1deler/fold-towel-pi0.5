# Fold towel π0.5 LoRA 微调准备

这是独立于已完成ACT的π0.5实验，不覆盖ACT代码、数据或模型。目标仓库为`git@github.com:k1deler/fold-towel-.git`，名称末尾有连字符；旧ACT仓库`fold-towel`继续保留。

## 数据与区别

- 复用51段fold_towel原录制的已校验转换数据，28,271帧，30Hz，两路480×640 RGB JPEG。保持只读，不重新上传2.61GB录制。
- 主臂状态是物理监督目标，从臂状态是观测；目标从观测时刻起`leader(t+k)`，offset0。不是现有旧π0.5适配器的`future_follower_absolute`标签。
- 保持与ACT同一组40段训练／11段验证，防止相邻帧跨集合；只用训练窗口拟合π0.5分位数归一化。ACT的mean/std或checkpoint不能直接当π0.5归一化或初始化。
- π0.5初始候选50步、内部32维（14物理维＋补零）、224×224保比例填边、prompt/state离散token200长度。缺少前视相机用黑图＋false掩码；只使用真实左右腕视角。
- 物理动作采用12关节相对同一次从臂观测的增量，夹爪绝对[0,1]。这与旧ACT“14维都做增量”不同，明确由输入输出适配器互逆，不能复用错误的夹爪还原。
- YAM数值保持原关节符号和夹爪尺度，**不套用ALOHA关节翻转或夹爪机械换算**。

## 模型与资源

源码为[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)固定commit`215abfb217dbac7d5f1273282331b9b1866c0479`，基础权重`pi05_base/params`。使用官方π0.5，不用`pi0_aloha_towel`冒充π0.5折毛巾模型。

初始配置在`configs/pi05_lora.json`：LoRA的Gemma2B和300M动作专家、batch4、10000步、warmup500、peak_lr5e-5、seed42、单卡FSDP1、wandb关闭。它是待小训练验证的候选，不是已调好的最优参数。官方资源说明给出LoRA与全量微调不同显存要求，不能直接照ACT分配资源。[OpenPI官方说明](https://github.com/Physical-Intelligence/openpi#requirements)

## 当前操作范围

用户已在CPU准备通过后明确请求开始训练。后台`scripts/train_pipeline.py`已启动，检测到一张空闲GPU后自动完成smoke→train；当前仍为排队状态，不能称为反向训练已开始。该管线只管理自己的子进程，有有限等待／执行时间，不触碰ACT和机器人。

1. 检查本地终端和训练服务器SSH。
2. 只读确认ACT结果、数据SHA、OpenPI版本、基础权重目录和Python/JAX环境。
3. 复制固定OpenPI代码到新实验`vendor/openpi`，不修改共享源码和环境。
4. 运行CPU适配器单元测试，准备训练集统计，并验证真实数据经过原OpenPI归一化／resize／tokenize／padding／DataLoader后的合同。
5. 保留后续显式GPU小训练与正式微调入口，**准备命令不自动启动GPU训练或机器人**。

实际检查结果见`reports/`，`preflight.json`只表示CPU数据管线通过，不表示12GB基础权重已经成功加载或反向更新。`gpu_smoke_passed.json`只有运行两步真实微调并产生checkpoint后才生成。最终是否能折叠毛巾需要单独的离线与闭环验证。

## 目录

| 路径 | 作用 |
|---|---|
| `fold_pi05/dataset.py` | 只读mmap数据、完整动作窗口、左右相机与动作互逆转换 |
| `fold_pi05/config.py` | 固定π0.5配置，进程内临时数据工厂；退出后恢复上游函数 |
| `fold_pi05/prepare.py` | 复核ACT数据哈希和相同episode划分，拟合训练集分位数统计 |
| `fold_pi05/preflight.py` | 真实数据、原OpenPI transforms／DataLoader的CPU验证 |
| `fold_pi05/train.py` | 显式GPU烟雾训练／正式微调，要求通过凭证、单张空闲卡，不抢占其他作业 |
| `fold_pi05/evaluate.py` | 真实微调后，在保留验证窗口上计算物理关节、夹爪、保持基线和越界率；不连接设备 |
| `scripts/prepare_server.sh` | 默认CPU准备；不构造实时设备 |
| `scripts/gpu_experiment.sh` | 后续现场明确启动的离线GPU计算；与机器人无连接 |
| `tests` | 目标源、边界、划分、夹爪还原、图像mask和官方动作mask一致性 |

## 训练服务器

服务器：`zhaoruiqi_yizu@101.132.150.51:14802`。新实验：`~/output/personal/k1deler/fold_towel_pi05_20261102`。ACT数据和结果继续留在`~/output/personal/k1deler/fold_towel_20261101`。

初期只读复用已存在`~/output/envs/yam-openpi/bin/python`，实际使用新实验固定源码副本。环境不是完全独立安装，版本和来源必须记录，不升级现有共享依赖。基础权重只读引用`~/output/data/pretrained/pi05_base/params`。

```bash
# 首次在新目录保存源码快照；复用指定机器已安装环境：
bash scripts/bootstrap_sources.sh

# 在新实验的code目录执行，不分配GPU：
bash scripts/prepare_server.sh

# 后续确认一张空闲GPU后，显式两步真实π0.5小训练：
bash scripts/gpu_experiment.sh smoke GPU_INDEX

# 仅小训练通过后才可开始正式微调：
bash scripts/gpu_experiment.sh train GPU_INDEX
```

GPU_INDEX替换成检查确认的空闲卡号。当前入口不允许共享忙卡，也不结束任何已有作业；不将“有剩余显存”误称为“空闲”。程序不会连接机械臂、实时相机、CAN或ACT硬件服务。

## Git与来源

代码通过用户指定SSH仓库保存。认证材料只在workspace的`work`用于登录，不放这里；Git排除原录制、JPEG存储、基础／训练权重、环境、cache和checkpoint。

OpenPI及其基础模型按官方许可使用；完整第三方代码不去署名，项目新增适配器和检查记录单独说明，见`THIRD_PARTY.md`。不因微调而把π0.5架构称为个人原创。
