# 透镜与光阑背景压制实验实现说明

日期：2026-10-08。

本次实现提供完整可执行的光学扫描、损失权重扫描和高分辨率复验流程。
现有 `rgb_decolorization_experiment.py` 的正式架构和七种历史损失变体保留。
本文件记录工程实现及验证边界，不将 smoke 结果解释为正式实验发现。

## 光学模型与坐标

1. 无滤波路径保留单 DOE 50 mm 和双 DOE 25 mm + 25 mm 自由传播。
2. 4f relay 采用标准物面到 L1、L1 到傅里叶面、傅里叶面到 L2、L2 到像面均为 f 的几何。
   使用两次按实际采样缩放的单位酉傅里叶变换；第二次变换还原物理像面，显式配准偶数网格的倒像。
   无光阑 relay 在数值误差范围内恢复物面场，不把频率坐标当成输出坐标。
3. 默认 relay 物面为原有自由传播像面。单 DOE 在 DOE 后 50 mm，双 DOE 在第二片后 25 mm。
   relay 再增加 4f 光路长度。`--relay-object doe` 将傅里叶分支物面改为最后一片 DOE。
   直接 DOE relay 是单独的架构测试：理想无滤波 relay 对单片纯相位 DOE 恢复均匀孔径强度，
   不能将该路径当作原先自由传播成像的等价替代。
4. 透镜面有限孔径分支采用 relay 的逐段角谱传播，在两片透镜面分别乘有限圆孔径与薄透镜二次相位。
   两片透镜共用固定半径和焦距，保存各透镜面前后的能量。透镜相位的最大局部空间频率
   `r_lens/(lambda_min*f)` 必须低于 Nyquist，避免未采样二次相位产生伪结果。
   `forward_rgb_lens_aperture()` 另外提供 DOE 与薄透镜共面、输出后焦面的解析模型；
   它取消输入二次相位、使用缩放 FFT，再通过含像素面积因子的复场插值映射探测坐标。
   第一阶段直接后焦面的完整 2.56 mm 输出视场不足，因此正式扫描选择 relay 像面分支。
   透镜分支的傅里叶面采用角谱传播的原始物理像素间距，理想无限透镜光阑分支采用缩放焦面 FFT 间距。
5. 傅里叶面采样为 `dx_F = lambda*f/(N_work*dx)`，光阑物理半径为 `r = lambda_ref*f*fc`。
   实际硬件 RGB 共用 532 nm 定义的物理半径和物理软边，三色实际截止频率为 `r/(lambda*f)`。
   接口也允许测试固定频率的理想通道掩模，但实验入口使用共享物理半径。
6. 傅里叶光阑软边采用 sigmoid，硬边复验采用二值掩模。通过第二次焦面变换保留相干边缘衍射。
   探测光阑是物理输出面的独立圆形孔径，额外保存裁剪前指标。

## 固定参数与扫描定义

DOE 半径 1 mm、共享高度上限 1.4 µm、488/532/633 nm 等功率、目标 circle/cross/ring50。
透镜扫描 50/100 mm 焦距和 0.5/0.75/1.0/1.25 mm 半径。
傅里叶截止为目标频谱半径的 60/80/100/120%；环形内半径为 DC 参考半径的 0/5/10/20%。
探测半径为目标包络的 1/1.25/1.5/2 倍，包络包括信号与过渡区。

为使“目标频谱半径”和“DC 半径”可复现，频谱定义为目标振幅 `sqrt(I_target)` 的频谱；
主半径包含其 95% 能量，DC 参考半径包含 10% 能量且至少为一个频率采样间隔。
这些定义及数值写入扫描清单和运行配置。光阑参数在 DOE 优化期间固定。
阶段之间继承物理半径，而不是在精网格重新定义一块不同的光阑。

第一阶段粗/精均为 128²/20 µm，40/80 次迭代，种子 7/17；先固定
`full_bg_absolute_peak` 扫描光学，再对最多三个候选扫描四种损失变体和规定的权重组合。
权重扫描严格限制为每个目标/架构最多三个入选光学配置各 55 组，不额外重复扫描每个分支代表；
每轮都保留同目标/架构/种子的 `full` 无滤波基线作为对照。
第二阶段 256²/10 µm → 512²/5 µm，120/400 次迭代，种子 7/17/27；
复验三个候选、各分支最佳代表及最佳无滤波损失组合。相同种子采用相同随机初始化和 schedule。
不满足 Nyquist 或视场要求的配置记录为采样拒绝，不作成功/失败的光学结论。
第一阶段 cross 的部分较大截止超过可表示频率；50 mm 透镜半径 0.75/1.0/1.25 mm
与 100 mm 透镜半径 1.25 mm 的二次相位也超过频率窗口，这些配置拒绝运行。

## 损失与功率

新增项为背景/ROI 能量比、背景平滑峰值/ROI 能量比和逐波长 ROI 效率下限的平方 hinge。
背景平滑峰值采用八阶 p-norm 近似，并除以同一波长的 ROI 能量；峰值计算先缩放再取范数，
以防小强度下溢、大强度溢出和零背景的不有限梯度。两个背景比在整体场强缩放时不变；
效率下限惩罚低于基线 90% 的通道，避免单靠整体衰减获得较低背景指标。
其峰值项为 `mean_lambda(||M_bg * I_lambda/P_lambda||_8 / (sum(M_signal * I_lambda/P_lambda) + eps))`，
RGB 波长按输入功率权重求平均。
效率项只有 `full_bg_constrained` 开启，缺少基线效率时立即报错。
新项采用 RGB schedule，前期关闭、中期线性增加，原有背景、孔洞 schedule 继续保留。

当前前向输入为单位振幅孔径，离散入射功率 `P_lambda` 等于孔径像素数，
RGB 相对功率仍为 1:1:1。新损失和评价使用这个真实离散功率，而不是把数千个单位振幅像素误当作总功率 1。
粗阶段和精阶段分别读取同种子、同网格基线的 ROI 效率，并各自计算 90% 下限。
绝对效率为 `sum(M_signal*abs(U)^2)/P_lambda`。报告中的背景峰值、P99 和均值单位为
“入射功率比例/输出像素”；它们只与同阶段同网格基线比较，跨网格不能直接视为功率密度。
背景/ROI 比采用加权掩模积分；P99 在背景掩模大于 0.5 的像素上统计。

## 验收与报告

逐目标、架构、种子，与无滤波 `full` 比较：背景/ROI 均值不超过基线 +0.005；
最差波长目标相关不低于基线 -0.05；每个 RGB 通道绝对效率至少为基线的 90%；
最大 RGB 组件质心漂移增量不超过 1 µm；圆环最大组件孔洞泄漏不超过基线 1.25 倍。
每个种子通过才标记配置跨种子通过，不能用平均效率掩盖一个失效通道。

满足约束者按背景比下降、P99 下降、绝对效率、共同尺度 ROI Pearson、能量 CV 排序。
候选分别关注背景下降、RGB 折衷和目标形状，最多三个不同光学配置。
若全部不通过，输出多目标 Pareto 前沿，并将复验候选标记为 Pareto fallback。
soft 和 hard 的验收分别保存，正式汇总与排序使用 hard 结果。

每个运行根目录有 `config.json`、`result.json`、粗阶段历史；`soft/` 和 `hard/` 分别有
三色强度、RGB 图、目标掩模、DOE 高度、损失历史、傅里叶面图、光阑掩模与指标。
结果显式包含 `optical_config`、`stop_throughput`、`background_metrics`、`acceptance_status`。
透过率区分透镜瞳孔、傅里叶光阑、探测孔径及计算视场裁剪；单位酉频谱能量是物面像素等价能量，
不能未经面积转换当作傅里叶面真实功率密度。

各阶段自动生成 CSV、JSON、候选清单、背景/效率图和英文简洁报告。
真实背景还可能包含散射、反射、加工误差、偏振、吸收和暗电流；这些未包含于当前标量理想模型。
只有背景、效率和形状同时满足约束，且不是单纯探测裁剪，才可解释为有效数值光学背景压制。

## 运行命令

```powershell
python src/rgb_background_filter_experiment.py --dry-run
python src/rgb_background_filter_experiment.py --stage screen
python src/rgb_background_filter_experiment.py --stage weights
python src/rgb_background_filter_experiment.py --stage validate
```

`--stage all` 连续执行。`--resume` 跳过完全相同配置的已完成运行；配置不匹配会拒绝恢复，避免混合实验。
正式扫描可能涉及数百至数千次运行。建议先看 `scan_manifest.json` 的采样拒绝，再启动正式计算。

小规模端到端验证命令：

```powershell
python src/rgb_background_filter_experiment.py --stage all --smoke `
  --targets circle --architectures single_doe two_doe_planes `
  --screen-seeds 7 --validation-seeds 7 `
  --screen-coarse-iterations 2 --screen-fine-iterations 3 `
  --validation-coarse-grid 128 --validation-coarse-pitch-um 20 `
  --validation-fine-grid 128 --validation-fine-pitch-um 20 `
  --validation-coarse-iterations 2 --validation-fine-iterations 3 `
  --device cpu --output-dir results/rgb_background_filter_smoke
```

smoke 的参数故意缩小，仅检验完整流程和非有限值防护。正式实验需要去掉 `--smoke` 并使用默认分辨率/迭代/种子。

## 验证记录

- `python -m pytest -q`：44 项测试通过，包括旧实验回归、4f 恢复、RGB 批处理、
  圆形光阑单调透过、软/硬维度与中心、软边参数梯度、零背景峰值梯度、有限透镜前后能量、
  效率下限、圆环硬约束、跨种子候选筛选、恢复配置检查和真实 JSON/数组持久化。
- 最终版本的 CPU 缩小三阶段测试位于 `results/rgb_background_filter_verified_smoke/`：
  circle、single_doe、种子 7、128²/20 µm、2/3 次迭代；screen 6 次、weights 8 次、validate 6 次，
  共 20 次运行，均保存 soft/hard 输出与验收报告。图像和 Pareto 图已检查。
- 双 DOE 圆环的非零中心遮挡、孔洞验收与恢复配置通过实际数值集成测试；
  早期双架构及 cross/ring50 图像 smoke 输出另存在结果目录中。
  最终交付使用上述 `verified_smoke` 的 schema_version=2 数据，早期结果不作为当前模型的正式结论。
- 新增峰值/ROI 定义后的 CUDA 三阶段 smoke 位于 `results/rgb_background_filter_gpu_smoke_peak_roi/`：
  circle、single_doe、种子 7，缩小迭代，仅验证 CUDA 前向、反向、结果保存和验收流程；不作科学结论。
- 扫描清单位于 `results/rgb_background_filter_manifest/scan_manifest.json`。
  circle/ring50 各 33 个配置，其中 29 个采样有效；cross 33 个，其中 19 个有效。
  每个入选光学配置有 55 种损失权重组合。
- 最终缩小样例中，符合全部条件的配置来自探测器空间选择；透镜和傅里叶分支未通过。
  这验证了程序能够区分背景下降与约束通过，不代表正式优化后仍然没有有效光学候选。
- 当前 RTX 5060 Ti/CUDA 正式运行位于 `results/rgb_background_filter_2026-10-10_peakroi_formal/`。
  低分辨率筛选已完成 320 次有效运行，另有 88 项采样拒绝；18 个入选记录目前均为 Pareto fallback。
  权重扫描正在运行，高分辨率三种子复验尚未完成；当前筛选指标不能替代最终 hard-stop 结论。
