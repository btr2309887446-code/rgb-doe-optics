# 多波长 DOE 数值设计

本项目研究：给定一个目标强度图案，能否设计衍射器件及配套光学系统，使不同波长在同一物理观察面形成形状、位置和尺寸接近的图案。每个新目标允许重新设计系统；设计完成后，各波长共用同一套器件与几何配置。

目前包含共享材料高度的 RGB DOE 设计、Fourier 光学反演与自由复屏研究，以及 400–700 nm 的多尺度图案联合设计。已有工作是有限目标和离散波长上的数值证据。

## 可见光波段研究入口

最新整理见[可见光波段研究说明](docs/visible_band_research.md)。2026 年 10 月 20 日实验以 400–700 nm 为设计波段、1 nm 为验证步长，另做 380–780 nm 压力测试：圆环控制组通过预设双门槛，多尺度目标尚未通过。

| 研究部分 | 内容与入口 |
|---|---|
| 共享高度 RGB DOE | 488/532/633 nm，材料色散、单/双 DOE、颜色一致性与背景折衷；见[实验指南](docs/experiment_guide.md) |
| Fourier 自由屏 | 450/532/633 nm 的屏函数自由度与训练外扫描；见 [Step 11 说明](多波长%20Fourier%20光学/fourier_optics_step11_README.md) |
| 可见光多尺度设计 | 34 个初始训练点、二阶/三阶光谱复屏、自适应补点；见[实验结果与数据索引](多波长%20Fourier%20光学/step11_visible_multiscale_20261020_full/README.md) |

仓库保存研究代码、测试、说明、指标、图表与小型最终设计文件；PPT 和本地汇报交付目录不纳入上传。

## 快速开始

需要 Python 3.10 或更新版本。先建立虚拟环境并安装依赖：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需 CUDA，请先从 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)选择与操作系统、Python 和 CUDA 版本相匹配的安装命令，再安装其余依赖。实验脚本默认使用可用的 CUDA GPU；传入 `--cpu` 可强制使用 CPU。

运行快速自检和测试：

```powershell
py -3 src/run_experiments.py --self-test --n-grid 32 --cpu
py -3 -m unittest discover -s tests -p "test_*.py"
```

一个小规模优化示例：

```powershell
py -3 src/run_experiments.py --iterations 50 --output-dir results/example --cpu
```

## 项目结构

```text
src/                       传播模型、材料、损失、优化器与实验入口
tests/                     数值模型、损失和实验工具测试
phase1_baseline/           单缝/多缝基线模拟
phase2_multiorder/         多级次设计模拟
report/                    LaTeX 报告、PDF、说明和报告图表
references/                引用文献索引
docs/experiment_guide.md   完整实验命令与参数指南
docs/visible_band_research.md  可见光多尺度研究整理与复算说明
多波长 Fourier 光学/       Fourier 反演、屏函数与可见光波段实验
chromatic_doe_study/        固定理想焦面上的共享材料高度研究
```

原 RGB 实验运行生成的数据默认写入 `results/`，该目录不纳入源码归档。Fourier 研究中已归档的结果及本次可见光研究的小型最终设计文件单独保留；每份约 100 MiB 的可见光逐波长强度数组和优化检查点保留在本地，可由最终设计复算。下载的书籍/论文、缓存、PPT 和临时文件不上传。`report/rgb_doe_honors_report.pdf` 保留前期共享高度研究报告，最新可见光结果以专门研究说明及实验目录为准。

## 模型范围

- 材料 DOE 分支以所有波长共享的物理高度图表示，并使用 S1813 折射率色散模型。
- 传播主要采用 FFT 角谱法；实验入口覆盖双波长/RGB、多图案、粗到细优化和透镜/光阑背景压制。
- Fourier 分支包含固定理想透镜与共同物理观察坐标下的自由复屏；其波长响应系数尚未映射为可加工的被动材料器件。
- 数值结果不等同于加工、实测或包含完整杂散光和材料误差的实验验证。各实验假设、命令与输出文件见[实验指南](docs/experiment_guide.md)及[报告](report/rgb_doe_honors_report.pdf)。

## 引用与许可

报告内含完整参考文献，`references/README.md` 提供核心文献索引。仓库中的原创代码、报告及自制图表按 MIT 许可证发布；未随仓库分发的第三方书籍、讲义与论文仍受各自版权条款约束。
