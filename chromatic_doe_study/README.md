# 多波长同图案 DOE 研究

本文件夹整理了前两项工作的结论，并实现了一项新的、受共享材料高度约束的多波长 DOE 逆向设计。完整讨论见 [研究报告](report.md)，程序和实际输出均保存在此文件夹。

## 复现

依赖 Python、NumPy、Matplotlib 和 PyTorch。PowerShell 中从课题目录执行：

    python -m pip install -r chromatic_doe_study/requirements.txt
    python chromatic_doe_study/chromatic_doe.py --self-test-only
    python chromatic_doe_study/chromatic_doe.py --output chromatic_doe_study/results --iterations 140 --starts 2

默认只用 CPU；可通过 `--device cuda` 请求 CUDA。程序每次运行都会覆盖指定输出目录中的结果文件。

## 已完成仿真

默认设计使用 488、532、633 nm 三个波长，同一张 S1813 高度浮雕和 32 级高度量化；532 nm 单色优化作为基线，RGB 联合优化从该解和两个随机起点开始。评价包括固定物理探测面上的图案相对 L2、ROI Pearson、入射能量归一化的 ROI 功率、16 级量化敏感性、10 nm 高度扰动和 488–633 nm 每 5 nm 的扫描。

自检覆盖匹配 DFT、解析伴随方向梯度的中心有限差分和离散 Parseval 能量关系。运行结果在 `results/summary.json`、`results/metrics.csv`、`results/spectral_scan.csv` 和下列图中：

- [三色图案](results/pattern_comparison.png)
- [高度图](results/height_maps.png)
- [单色/联合优化对照](results/baseline_vs_joint.png)
- [训练波段扫描](results/spectral_holdout.png)
- [优化历史](results/optimization_history.png)

完整的物理模型、结果解释、适用范围和文献引用在 [report.md](report.md)。
