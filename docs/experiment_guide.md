# 实验脚本与复现指南

本指南汇总项目的实验入口、参数和运行示例。命令默认从仓库根目录执行；生成文件写入 `results/`，该目录以及历史运行数据不会随源码仓库发布。GPU/CUDA 环境请先按根目录 README 中的 PyTorch 指引安装。

## 最小双波长示例

本项目的第一步不是直接追求“任意图案”，而是验证一个最小问题：

> 设计一个静态相位型 DOE，使 532 nm 和 633 nm 光在同一个物理观察面上都形成位于相同位置的两个光斑。

程序同时给出两个结果：

1. **单波长基线**：只用 532 nm 优化，随后测试 532 nm 和 633 nm；
2. **双波长设计**：一个共享的物理高度 `h(x,y)` 同时优化两个波长。

这样能够直接说明“普通单波长设计为什么不够”和“多波长联合优化能改善什么”。

## 运行

在项目根目录打开 PowerShell：

```powershell
py -3 src/two_spot_multicolor.py
```

如果只想先快速检查：

```powershell
py -3 src/two_spot_multicolor.py --iterations 50
```

默认输出在 `results/two_spot/`，包括：

- `target_two_spots.png`：目标两个光斑；
- `baseline_results.png`：单波长 DOE 在两个波长下的结果；
- `multicolor_results.png`：共享高度的双波长 DOE 结果；
- `baseline_height.png`、`multicolor_height.png`：DOE 高度分布；
- `loss_history.png`：优化曲线；
- `two_spot_design.npz`：可用 NumPy 读取的原始数据；
- `metadata.json`：参数和相关系数。

## 当前数值模型

- 传播：标量角谱法；
- DOE：薄、无吸收、相位型透射元件；
- 采样：`256 x 256`，像素间距 `10 µm`；
- 光路：均匀入射场 → 有限孔径 S1813 相位 DOE → 空气中角谱传播 → DOE 后 `5 cm` 的物理像平面；
- 波长：`532 nm` 和 `633 nm`；
- 最大高度：`1.4 µm`；
- 光刻胶：Shipley S1813，`n(532 nm)=1.6482`，`n(633 nm)=1.6347`；
- 目标：在 `x = ±0.30 mm, y = 0` 处的两个高斯光斑；
- 优化变量：一个共享的物理高度图，而不是两张独立相位图。

## 结果应该如何解释

程序用归一化强度与目标强度的相关系数作为第一项指标。双波长 DOE 不一定在两个波长下都比单波长基线的 532 nm 结果更好；合理的现象是：

- 单波长基线在 532 nm 下较好，在 633 nm 下明显变差；
- 双波长设计牺牲一部分 532 nm 性能，换取两个波长都可接受；
- 波长差、目标光斑间距、器件像素数和高度量化会改变结果。

## 下一步

1. 重复运行并保存随机种子固定的基线结果。
2. 扫描 `--iterations`、光斑间距和波长间隔。
3. 把连续高度量化为 2、4、8、16 级，评估制造可行性。
4. 用多个随机种子重复优化，报告均值和标准差。
5. 加入高度量化、制造误差和背景能量指标，评估器件可制造性。
6. 如实验系统确实包含透镜，再单独建立“透镜—像平面”模型，输入实测物距、像距和焦距。


## 像平面模型说明

当前“像平面”指 DOE 后方 `z = 5 cm` 的共用物理观察面，而不是透镜后焦平面。由于两个波长使用同一个 `x,y` 网格和同一个传播距离，目标中的光斑位置是物理坐标位置，不是仅仅相同的衍射角。传播实现使用输入 `ifftshift`、本征 FFT 顺序传递函数和输出 `fftshift`，用于保持中心坐标不发生数组环绕。

# 双波长/多波长相同双光斑 DOE：模块化实验管线

当前实验管线已扩展为四个可复用模块：

- `src/forward_models.py`：居中 FFT 角谱传播、自由传播、薄透镜像平面和零填充；
- `src/materials.py`：S1813 两点 Cauchy 色散模型；
- `src/targets.py`：软边圆形双光斑及信号/过渡/背景区域；
- `src/losses.py`、`src/metrics.py`：联合损失和定量评价；
- `src/optimizers.py`：Adam 与 MRAF/OMRAF 风格优化；DBS 仅保留为旧实验工具；
- `src/run_experiments.py`：统一实验入口。

## 基础自检

```powershell
py -3 src/run_experiments.py --self-test --n-grid 32 --cpu
py -3 -m unittest discover -s tests -p "test_*.py"
```

自检会验证：中心坐标恒等传播、传播能量守恒以及薄透镜方程。

## 四波长 Adam

```powershell
py -3 src/run_experiments.py `
  --wavelengths 488 532 633 660 `
  --algorithm adam `
  --iterations 600 `
  --n-grid 256 `
  --pixel-pitch-um 10 `
  --radius-um 40 `
  --edge-width-um 15 `
  --validate-band `
  --output-dir results/four_wavelength_adam `
  --cpu
```

488 nm 和 660 nm 的折射率是由 S1813 的 532/633 nm 锚点外推得到，结果 JSON 会标记 `extrapolated=true`，不能把外推值写成直接测量值。

## 课程式训练

先在 532/633 nm 预训练，再继续加入 488/660 nm：

```powershell
py -3 src/run_experiments.py `
  --curriculum `
  --curriculum-pre-iterations 300 `
  --iterations 600 `
  --output-dir results/curriculum_four_wavelength `
  --cpu
```

## 算法比较

```powershell
py -3 src/run_experiments.py `
  --algorithm compare `
  --iterations 200 `
  --dbs-sweeps 2 `
  --dbs-max-pixels 200 `
  --output-dir results/algorithm_compare_smoke `
  --cpu
```

正式 DBS 不设置 `--dbs-max-pixels`，但计算量会显著增加；建议先在 64 或 128 网格上完成算法初筛，再在 256/512 网格复核。

## 薄透镜像平面

```powershell
py -3 src/run_experiments.py `
  --model thin_lens `
  --focal-length-mm 50 `
  --magnification 1 `
  --iterations 600 `
  --output-dir results/thin_lens_adam `
  --cpu
```

程序由

\[
\frac{1}{f}=\frac{1}{s_o}+\frac{1}{s_i}
\]

自动计算物距和像距。透镜模型与自由传播模型共享同一目标、损失和评价指标。

## 结果文件

每个实验目录包括：

- `summary.json`：完整配置、材料信息、损失历史和指标；
- `*_seed*_height.npy`：物理高度图，单位米；
- `*.png`：各波长输出图；
- `heights.npz`：高度图归档。

旧目录 `results/optimization_sweep`、`results/target_processing` 和 `results/target_parameter_scan*` 使用旧的单段脚本与旧目标/常数折射率，只能作为历史记录。最终报告应使用新的模块化入口重新生成结果。
## 参数扫描

重新生成软边目标和背景权重扫描：

```powershell
py -3 src/scan_modular.py --kind target --radii 30 40 50 60 --edge-widths 10 15 20 25 --iterations 300 --output-dir results/scan_targets --cpu
py -3 src/scan_modular.py --kind weights --background-weights 0.05 0.10 0.25 0.50 --iterations 300 --output-dir results/scan_weights --cpu
```

扫描结果按最差波长相关系数、平均背景比例和 ROI 效率排序，并输出 `pareto.png`。

## 多目标泛化性比较

泛化实验让每个目标独立优化，只比较 Adam 与 MRAF/OMRAF 风格优化。
此前的快速 DBS 结果未达到可用质量，因此不再纳入当前比较。默认目标包括
软边双圆斑、双高斯、双环、非对称双斑和三角形三圆斑。

先运行小规模验收：

```powershell
py -3 src/generalization_experiment.py `
  --n-grid 128 `
  --pixel-pitch-um 10 `
  --aperture-radius-um 500 `
  --iterations 2 `
  --validation-iterations 2 `
  --targets all `
  --methods adam mraf `
  --seeds 7 `
  --quick `
  --output-dir results/generalization_smoke `
  --cpu
```

正式的第一轮筛选：

```powershell
py -3 src/generalization_experiment.py `
  --n-grid 256 `
  --iterations 100 `
  --targets all `
  --methods adam mraf `
  --seeds 7 `
  --output-dir results/generalization_coarse `
  --cpu
```

脚本会先在固定候选参数网格中为每个目标和方法选出配置，再使用指定种子
复算。每个运行目录保存 `target.png`、各波长 `output.png`、`height.png`、
`metrics.json`、`loss_history.json` 和 `height.npy`；总目录保存
`summary.json`、`method_comparison.csv`、相关性/背景热图和 Pareto 图。

指标中的组件能量比例会跟随目标本身的期望比例，因此非对称双斑不会被错误地
强制成等能量；三角形目标则会报告三个独立光斑的能量、质心、FWHM 和圆形度。
已有 `results/generalization_300iter` 中的 DBS 文件仅作为历史记录保留；
重新运行泛化实验时不再生成 DBS 结果。

## 圆斑/圆环粗到细优化

圆环实验使用正确覆盖 `1 mm` 半径 DOE 的物理视场，并将粗网格高度图
继承到细网格。默认设置为：

- 粗阶段：`256×256`、`10 µm`、`pad_factor=2`；
- 精阶段：`512×512`、`5 µm`、`pad_factor=2`；
- 波长：`532/633 nm`；
- 目标：半径 `50 µm`、宽度 `8 µm` 和半径 `60 µm`、宽度 `10 µm` 两种圆环；
- 方法：Adam、MRAF；
- 输出：线性像面、固定 `-40 dB` 像面、径向强度曲线、圆环内孔指标和高度图。

先运行有效采样的小规模检查：

```powershell
py -3 src/ring_optimization_experiment.py `
  --coarse-grid 128 `
  --coarse-pitch-um 20 `
  --fine-grid 256 `
  --fine-pitch-um 10 `
  --coarse-single-iterations 5 `
  --coarse-joint-iterations 5 `
  --fine-iterations 5 `
  --targets circle ring50 `
  --methods adam mraf `
  --seeds 7 `
  --quick `
  --output-dir results/ring_optimization_smoke `
  --cpu
```

正式运行：

```powershell
py -3 src/ring_optimization_experiment.py `
  --output-dir results/ring_optimization `
  --coarse-single-iterations 300 `
  --coarse-joint-iterations 300 `
  --fine-iterations 800 `
  --targets circle ring50 ring60 `
  --methods adam mraf `
  --seeds 7 17 27 `
  --cpu
```

程序会在采样不足时直接报错。每个最终运行目录包含
`output_linear.png`、`output_db.png`、`radial_profile.png`、
`radial_profile.csv`、`metrics.json`、`loss.png`、`height.png` 和
`result.json`。圆环结果重点查看 `mean_center_to_ring_ratio`、
`mean_hole_leakage_fraction` 和 `mean_ring_radius_error_m`。

## RGB 多图案泛化实验

RGB 泛化入口支持 `488/532/633 nm` 三波长，以及双圆斑、双圆环、双方形、
双十字、三角形三圆斑、实心三角形和不对称双圆斑。它使用共享的粗到细
传播模型，并记录三波长质心漂移和圆环半径漂移。

先运行低成本验收：

```powershell
py -3 src/rgb_generalization_experiment.py `
  --coarse-grid 32 `
  --coarse-pitch-um 100 `
  --fine-grid 32 `
  --fine-pitch-um 100 `
  --coarse-iterations 1 `
  --fine-iterations 1 `
  --targets circle ring50 ring60 square cross triangle_three solid_triangle asymmetric `
  --methods adam `
  --seeds 7 `
  --position-weights 0.10 `
  --output-dir results/rgb_generalization_targets_smoke `
  --cpu
```

正式 RGB 实验：

```powershell
py -3 src/rgb_generalization_experiment.py `
  --coarse-grid 256 `
  --coarse-pitch-um 10 `
  --fine-grid 512 `
  --fine-pitch-um 5 `
  --coarse-iterations 300 `
  --fine-iterations 1200 `
  --targets circle ring50 ring60 square cross triangle_three solid_triangle asymmetric `
  --methods adam mraf `
  --seeds 7 17 27 `
  --position-weights 0 0.10 0.25 `
  --geometry-weights 0.10 `
  --curriculum `
  --output-dir results/rgb_generalization `
  --cpu
```

正式结果中重点查看每个运行目录的 `output_db.png`、`metrics.json` 和
`result.json`；总目录保存 `summary.json`、`summary.csv` 和
`experiment_config.json`。当前版本的几何权重接口已保留，方形、十字和三角形
的主要形状约束仍由目标区域振幅损失承担。

## RGB 跨波长平移与多器件比较

`src/chromatic_shift_experiment.py` 针对双圆斑、双十字、R50 双圆环比较：

- 单 DOE 后自由传播；
- DOE 与共面理想薄透镜后传播到焦平面；
- 两张独立 DOE 相位面，中间自由传播。

RGB 三个波长通过一个 batch 维度并行计算。传播算子采用角谱传递函数在
傅里叶域的对角形式，以 GPU FFT 执行；不构造庞大的稠密传播矩阵。目标先以
4 倍子像素采样，再扫描 logit 边缘锐化增益；损失同时包含目标边缘梯度误差、
目标位置误差和超过 5 µm 的跨波长质心漂移惩罚。先运行短时 CUDA 验收：

```powershell
py -3 src/chromatic_shift_experiment.py `
  --coarse-grid 64 --coarse-pitch-um 40 `
  --fine-grid 64 --fine-pitch-um 40 `
  --coarse-iterations 2 --fine-iterations 2 `
  --targets circle --seeds 7 `
  --output-dir results/chromatic_shift_smoke
```

正式比较使用相同 2.56 mm 视场的粗、细网格，并默认在每种目标/光路下用 3 个
随机种子验证：

```powershell
py -3 src/chromatic_shift_experiment.py `
  --coarse-grid 256 --coarse-pitch-um 10 `
  --fine-grid 512 --fine-pitch-um 5 `
  --coarse-iterations 120 --fine-iterations 400 `
  --targets circle cross ring50 `
  --architectures single_doe doe_fourier_lens two_doe_planes `
  --seeds 7 17 27 `
  --output-dir results/chromatic_shift_multiplane
```

程序默认使用可用 CUDA GPU；添加 `--cpu` 可强制 CPU。总目录生成 `image_plane_overview.png`、
`chromatic_shift_comparison.png`、`target_processing_comparison.png`、`summary.csv` 和
`summary.json`。各次运行保留 RGB 三个波长的线性/dB 像面图、伪彩色叠加图、指标、
损失历史和 DOE 高度。伪彩色叠加图对每个波长独立归一化，只用于观察相对位置，
不代表真实色度或绝对能量比。薄透镜按理想消色差薄透镜建模；实验结果不能直接
代表真实透镜材料的色差表现。

因为粗扫可能为不同光路选择不同的边缘锐化增益，完成优化后建议统一目标再评估，
避免图案定义差异干扰光路比较：

```powershell
py -3 src/analyze_chromatic_shift.py `
  --results-dir results/chromatic_shift_multiplane `
  --reference-sharpness 1.6
```

此复评生成 `common_reference_overview.png`、`common_reference_shift.png` 和
`common_reference_metrics.csv/json`，所有器件结构都对同一锐化目标计算指标。
若需要把 RGB 最大质心漂移压到 5 µm 以下，可对已保存高度图执行额外约束精修，
再用 `--prefer-refined` 生成统一目标对照图：

```powershell
py -3 src/refine_chromatic_shift.py `
  --results-dir results/chromatic_shift_multiplane `
  --iterations 120 --position-weight 2.0 --centroid-tolerance-um 3.0
py -3 src/analyze_chromatic_shift.py `
  --results-dir results/chromatic_shift_multiplane `
  --reference-sharpness 1.6 --prefer-refined
```

## 小网格稠密传播矩阵验证

用 `src/dense_propagation_experiment.py` 对比两类传播矩阵：直接离散 Fresnel 核，以及与现有补零/裁剪角谱 FFT 等价的稠密矩阵。默认使用 `32 x 32` 网格、固定 S1813 正弦相位光栅和 `488/532/633 nm`，不进行高度优化。

```powershell
py -3 src/dense_propagation_experiment.py
```

可通过 `--n-grid`、`--pixel-pitch-um`、`--distance-mm`、`--aperture-radius-um`、`--grating-period-um`、`--grating-height-um` 和 `--wavelengths-nm` 控制模型；`--device cpu` 可切换到 CPU。单个矩阵超过默认 `256 MiB` 时会在分配前报错。输出保存到 `results/dense_propagation/`，包括逐波长像面图、复场/线性强度数组、中心剖面、误差和耗时汇总。角谱稠密矩阵只适用于小网格验证：其存储量按网格边长的四次方增长，不代表大网格下比 FFT 更快。

## 传播收敛与 RGB 共同色标复评

`src/propagation_convergence_experiment.py` 固定 `320 µm` 物理视场，扫描
`16/24/32/48/64` 网格和 `488/532/633 nm`，比较角谱 FFT、等价角谱稠密矩阵和离散 Fresnel 矩阵；同时检查真实 `256×256/10 µm` 与
`512×512/5 µm` DOE 的孔径覆盖和能量守恒：

```powershell
py -3 src/propagation_convergence_experiment.py `
  --grid-sizes 16 24 32 48 64 `
  --wavelengths-nm 488 532 633 `
  --output-dir results/propagation_convergence `
  --device cuda
```

角谱稠密矩阵应在所有网格上复现 FFT；Fresnel 矩阵只作为近似模型对照，若误差不随采样稳定下降，应先检查核的坐标、边界和归一化，不要据此扩大 DOE 优化规模。

`src/rgb_imageplane_evaluation.py` 不重新优化高度图，而是读取已有
`results/chromatic_shift_multiplane/`，以等入射功率 `488:532:633=1:1:1` 重新生成：

- 每个波长独立峰值归一化的形状图；
- 共享峰值的线性色标和 `-40~0 dB` 色标图；
- 原始像面能量、ROI 绝对效率、背景绝对能量；
- 共同尺度与形状归一化下的跨波长像素相似度、组件质心漂移。

```powershell
py -3 src/rgb_imageplane_evaluation.py `
  --results-dir results/chromatic_shift_multiplane `
  --targets circle cross ring50 `
  --architectures single_doe doe_fourier_lens two_doe_planes `
  --seeds 7 17 27 `
  --power-weights 1 1 1 `
  --output-dir results/rgb_imageplane_evaluation
```

RGB 合成仅用于显示，明确标注为非 CIE 色度渲染。共同色标下的通道差异应结合 `energy_by_wavelength.csv` 和 `cross_wavelength_similarity.csv` 解读：若总能量不同但形状相似，优先加入波长能量平衡；若共同尺度下像素纹理仍不一致，下一步再加入跨波长逐像素、边缘和背景一致性损失。

## RGB 跨波长消色散损失消融

`src/rgb_decolorization_experiment.py` 在不使用 `doe_fourier_lens` 和 DBS 的前提下，比较 `single_doe` 与 `two_doe_planes`。实验固定 RGB 波长 `488/532/633 nm`、等入射功率 `1:1:1`，并对 `circle`、`cross`、`ring50` 分别运行 `baseline`、`pixel`、`edge`、`background`、`energy`、`component` 和 `full` 七种损失变体。每个运行都采用 `256→512` 粗到细高度继承，不读取历史 `drift_refined` 高度图。

先运行 CUDA smoke test：

```powershell
py -3 src/rgb_decolorization_experiment.py `
  --coarse-grid 64 --coarse-pitch-um 40 `
  --fine-grid 64 --fine-pitch-um 40 `
  --coarse-iterations 10 --fine-iterations 20 `
  --targets circle `
  --architectures single_doe two_doe_planes `
  --variants baseline pixel edge background energy component full `
  --seeds 7 --device cuda `
  --output-dir results/rgb_decolorization_smoke
```

正式实验命令：

```powershell
py -3 src/rgb_decolorization_experiment.py `
  --coarse-grid 256 --coarse-pitch-um 10 `
  --fine-grid 512 --fine-pitch-um 5 `
  --coarse-iterations 120 --fine-iterations 400 `
  --targets circle cross ring50 `
  --architectures single_doe two_doe_planes `
  --variants baseline pixel edge background energy component full `
  --seeds 7 17 27 --device cuda `
  --output-dir results/rgb_decolorization
```

总目录中的 `summary.csv`、`ablation_summary.csv`、`pixel_similarity_heatmap.png` 和 `absolute_energy_cv_heatmap.png` 分别用于比较共同色标像素相似度和 RGB 绝对能量变异；每次运行目录同时保存三色线性像面、共同色标/dB 图、目标掩膜、DOE 高度、损失历史和逐波长指标。`full` 只有在共同尺度 ROI Pearson 提升、RGB ROI 能量 CV 下降且最差波长相关系数没有明显恶化时，才能称为真正的像素级消色散；单纯降低三色总能量不算成功。

## 透镜与光阑背景压制实验

`src/rgb_background_filter_experiment.py` 是独立入口，不覆盖历史消色散实验。
新增绝对背景/ROI 比、稳定的背景 p-norm 峰值损失、基线 90% 的逐波长效率下限，
并支持透镜有限 NA、圆形/环形傅里叶光阑和独立探测孔径。

先生成采样检查清单，再依次运行光学扫描、候选权重扫描和完整验证：

```powershell
python src/rgb_background_filter_experiment.py --dry-run
python src/rgb_background_filter_experiment.py --stage screen --device auto
python src/rgb_background_filter_experiment.py --stage weights --device auto
python src/rgb_background_filter_experiment.py --stage validate --device auto
```

也可使用 `--stage all` 连续执行，或用 `--resume` 恢复配置一致的已完成运行。
默认第一阶段为 128²/20 µm、40/80 迭代、种子 7/17；第二阶段为
256²/10 µm → 512²/5 µm、120/400 迭代、种子 7/17/27。
每阶段均保留同目标、同架构、同种子的无滤波 `full` 基线。

默认 4f relay 的物面是现有自由传播后的像面，从而对已有 DOE 成像结果进行频率滤波。
`--relay-object doe` 可测试直接 DOE 成像，但纯相位单 DOE 的无滤波 1:1 relay
会恢复孔径强度，本身不能形成非均匀目标。有限透镜分支采用 relay 上的显式薄透镜相位、
有限孔径和逐段角谱传播；直接后焦面模型提供在 `forward_rgb_lens_aperture()` 中。
这些物理边界、光学位置和采样关系写入每次运行的 `config.json`。

RGB 共用固定物理光阑，按 532 nm 参考频率换算半径；因此 RGB 的实际截止频率不同。
超出 Nyquist 或计算视场的配置会被拒绝并记录，不能通过截断采样冒充有效光阑。
探测孔径结果单独标记为“探测器空间选择”，同时保存裁剪前后的绝对指标。

各阶段输出 `summary.json/csv`、`aggregate.csv`、`candidates.json`、Pareto 图和 `report.md`。
各运行输出软/硬光阑图、RGB common-scale/shape-normalized 图、傅里叶面图、能量透过率、
背景均值/最大值/99 分位值、ROI 绝对效率、圆环泄漏、质心和半径漂移，以及硬约束验收。
详细使用说明与建模限制见 `report/background_filter_implementation.md`。
