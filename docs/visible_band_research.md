# 可见光波段同图案成像研究整理

## 研究问题与已有工作的关系

给定一个目标强度图案，通过设计衍射器件及光学系统，能否使不同波长在同一物理观察面形成相同图案？每个目标可以重新设计；设计完成后，所有波长共用固定系统。评价先比较位置、尺寸和强度形状，亮度、背景与有效能量另行检查。

研究从共享材料高度的 RGB DOE 出发，再用 Fourier 光学拆分传播缩放、有限孔径和屏函数参数化的影响。最新实验把三色设计扩展到 400–700 nm，并用多尺度分辨率图检验复杂目标，圆环作为控制组。这里的设计波段取 400–700 nm；380–780 nm 单独作为压力测试范围，不与设计波段混为一谈。

## 2026 年 10 月 20 日保存的实验

- 固定理想 Fourier 透镜焦距 0.2 m；所有波长在共同物理观察网格计算，不靠输出图后处理缩放实现重合。
- 屏面 320×320；观察面 256×256，像素间距 15.625 µm，视场约 4 mm；孔径沿用 Step 11，孔径内有效像素 5,025。
- 多尺度目标含 160/240/360 µm 线宽与 0°/45°/90°方向，共九个区域及配准标记；按原函数抗锯齿生成，各单元等积分能量。
- 屏函数为 Chebyshev 波长复系数展开后逐波长归一化，比较二阶与三阶响应；分别有 30,150 和 40,200 个实参数。
- 初始训练点为 400–700 nm 每 10 nm，加 488/532/633 nm，共 34 点。每阶 3 个候选起点，每起点最多 400 次 L-BFGS-B 迭代；三阶由对应二阶候选零扩展初始化。
- 多尺度三阶模型另做两轮自适应补点，每轮加入 6 点，最终训练点为 46；圆环控制组保持 34 点，不补点。

损失为：

\[
L=0.7\,\mathrm{mean}(e_\lambda^2)
+0.3\,\mathrm{CVaR}_{\mathrm{worst}\ 10\%}(e_\lambda^2)
+0.1\,\mathrm{mean}\{\max(0,0.80-P_{\mathrm{ROI},\lambda})^2\}.
\]

其中 \(e_\lambda=\|I_\lambda-I_t\|_2/\|I_t\|_2\)。CVaR 使用分数边界样本保持最差 10% 的精确权重。ROI 是目标不低于峰值 1% 的像素；功率比例以屏面总透射功率为分母，不按有限观察窗功率重新归一化。

验收要求 **400–700 nm 全部 301 个 1 nm 采样点同时满足相对 L2 ≤ 0.10、ROI 功率比例 ≥ 0.80**。平均值满足阈值不能代替逐点通过。

## 已有结果

下表均采用相同的 400–700 nm、1 nm 验证网格，数值来自保存的逐波长结果。

| 目标 | 模型 | 最终训练点数 | 平均相对 L2 | 最差相对 L2 | 最低 ROI 比例 | 双门槛 |
|---|---|---:|---:|---:|---:|---|
| 多尺度 | 二阶 | 34 | 0.445771 | 0.453562 | 0.697048 | 未通过 |
| 多尺度 | 三阶及两轮补点 | 46 | 0.384106 | 0.395489 | 0.752307 | 未通过 |
| 圆环 | 二阶 | 34 | 0.078602 | 0.093790 | 0.957779 | 通过 |
| 圆环 | 三阶 | 34 | 0.011048 | 0.040341 | 0.991720 | 通过 |

多尺度三阶最终解的平均误差比二阶低约 13.8%，但同时增加了模型阶数、自适应训练点和优化预算，不能将该差值全部归因于阶数。圆环结果说明当前模型能拟合较简单目标；多尺度目标的误差仍大，尚不能据此确定是表达能力、局部优化、损失权重或其他约束单独造成，更不能作为不可达性证明。

在 380–780 nm 压力范围，三阶多尺度和圆环的最差误差分别为 0.790354 和 0.156820，均在 780 nm；不能将设计波段通过扩大表述为整个压力范围通过。分区 Pearson 较高只表示空间形状相关，不等于相对 L2 与能量门槛同时通过。

图表、P95、分区相关性与补点记录见[完整报告](../多波长%20Fourier%20光学/step11_visible_multiscale_20261020_full/report.md)；文件入口见[实验归档索引](../多波长%20Fourier%20光学/step11_visible_multiscale_20261020_full/README.md)。

## 运行与复算

从仓库根目录安装依赖并运行测试：

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m unittest discover -s tests -p "test_*.py"
py -3 "多波长 Fourier 光学/run_visible_multiscale_design.py" --check-only
```

需要重新优化时，指定一个不存在的新目录，原归档不会被覆盖：

```powershell
py -3 "多波长 Fourier 光学/run_visible_multiscale_design.py" --output "多波长 Fourier 光学/step11_visible_multiscale_new_run"
```

默认参数与保存实验一致。原实验运行中使用过检查点恢复；归档保留原始配置及来源哈希。整理时发现当前实验入口脚本的 SHA-256 与原 `provenance.json` 记录不同，而底层传播与优化模型文件的哈希一致；发布清单分别保留原记录与当前哈希，不覆盖实验来源记录。四份最终设计在 400/532/700 nm 的传播复算用于核对已保存指标，不宣称当前入口或不同运行环境会逐比特重现原优化过程。

最终 `design.npz` 已包含复系数、参数向量、孔径、目标、屏面与观察面坐标、训练和扫描波长。以下 Python 示例从仓库根目录复算某个已保存设计，无需重新优化：

```python
import sys
from pathlib import Path
import numpy as np

fourier = Path("多波长 Fourier 光学")
sys.path.insert(0, str(fourier.resolve()))
from fourier_optics_step11_free_screen_optimization import Model, Problem

run = fourier / "step11_visible_multiscale_20261020_full"
design = run / "multiscale/spectral_degree_3/design.npz"
with np.load(design, allow_pickle=False) as saved:
    problem = Problem(saved["pupil"], saved["target"], saved["training_nm"],
                      saved["band_nm"], grid_nm=400.0)
    model = Model(problem, "spectral", 3)
    # 只传播，不调用优化器；这里演示三个波长。
    metrics, images, _ = model.evaluate(saved["theta"], [400., 532., 700.],
                                       save_arrays=True)
for row in metrics:
    print(row["wavelength_nm"], row["relative_L2"], row["roi_power"])
```

将示例波长换成 `saved["scan_nm"]` 可重建全扫描强度；四份 401×256×256 的 float32 扫描数组各为 100.25 MiB，保留本地，不进入 Git。优化检查点也不上传。归档中的 JSON、CSV、图表、目标和最终设计足以查看结论并复算传播。

## 解释边界

这些结果使用标量、近轴、点采样的理想 Fourier 透镜模型。自由复屏允许局部振幅超过 1，其波长系数尚未对应具体材料或结构；ROI 比例是数值功率收集比例，不代表真实器件效率。密集离散扫描也不是连续波段的数学保证。前期 RGB 的 Pearson、不同实验的平均 L2/RMS L2 及不同 ROI 定义应分别解释。
