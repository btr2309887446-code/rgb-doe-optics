# 多波长 DOE 数值设计

本项目研究共享物理高度的衍射光学元件（DOE）如何在多个可见光波长下重建形状和位置接近的目标图案。数值模型以标量角谱传播为主，包含材料色散、单/双 DOE 光路、RGB 一致性损失、空间滤波实验及结果分析工具。

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
```

运行生成的数据默认写入 `results/`。原始实验结果、下载的书籍/论文、缓存和临时文件不属于此公开源码归档；可按实验指南重新生成结果。`report/rgb_doe_honors_report.pdf` 是当前研究报告，LaTeX 源文件及其图表也一并提供。

## 模型范围

- DOE 以所有波长共享的物理高度图表示，并使用 S1813 折射率色散模型。
- 传播主要采用 FFT 角谱法；实验入口覆盖双波长/RGB、多图案、粗到细优化和透镜/光阑背景压制。
- 数值结果不等同于加工、实测或包含完整杂散光和材料误差的实验验证。各实验假设、命令与输出文件见[实验指南](docs/experiment_guide.md)及[报告](report/rgb_doe_honors_report.pdf)。

## 引用与许可

报告内含完整参考文献，`references/README.md` 提供核心文献索引。仓库中的原创代码、报告及自制图表按 MIT 许可证发布；未随仓库分发的第三方书籍、讲义与论文仍受各自版权条款约束。
