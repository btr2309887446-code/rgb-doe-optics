# 多波长 Fourier 光学研究进度

已有 Step 1–9 的 NumPy/SciPy 数值模型、结果数据、图和阶段结论。仿真使用用户指定的 `SI100B_Project` Conda 环境；已检查该环境，当前没有 PyOptica，所以计算基于 NumPy/SciPy FFT 与插值流程。

## 阶段结果

| 阶段 | 当前结论 | 报告或数据 |
|---|---|---|
| 1. 任意屏 → Fourier 衍射 | 圆孔 Airy、相位光栅和任意复屏的数值结果与解析/离散 DFT 对照吻合 | [Step 1 报告](fourier_optics_step1_report.md) |
| 2. 任意复振幅 | 匹配 DFT 网格可逆；屏支持有限时，宽带目标在像素间可能出现显著误差 | [Step 2 报告](fourier_optics_step2_report.md) |
| 3. 任意强度、相位型屏 | GS 可近似三光斑、圆环和字母 A；结果受初值影响，不能把“任意”当成无条件精确可实现 | [Step 3 报告](fourier_optics_step3_report.md) |
| 4. 多波长坐标缩放 | 固定焦距时图案坐标随 λ 缩放；理想 (f\propto1/\lambda) 时三色强度图重合 | [Step 4 报告](fourier_optics_step4_report.md) |
| 5. 基本图案与有限孔径 | 复场 DFT 基完整正交；圆孔截断使模式耦合并形成 Airy 型有限光斑 | [Step 5 报告](fourier_optics_step5_report.md) |
| 6. 有限器件约束 | 相位量化、像素间距和 NA 扫描显示可达图案受硬件限制 | [Step 6 报告](fourier_optics_step6_report.md) |
| 7. 普通透镜下共享相位屏 | 幅度共识启发式使三色共同网格平均 L2 从 0.623 降至 0.610，最差值从 0.852 降至 0.811，但 532 nm 误差从 0.251 升至 0.371；不是全局优化保证 | [Step 7 脚本](fourier_optics_step7.py)、[数据](fourier_optics_step7_results.json) |
| 8. 熔融石英浮雕色散 | 在人为指定的理想 λf 补偿下，固定高度浮雕仍使相位屏随波长变化；450/633 nm 相对理想 532 nm 图案 L2 漂移为 0.178/0.227 | [Step 8 报告](fourier_optics_step8_report.md) |
| 9. 浮雕高度多色联合优化 | 连续高度三色联合优化使平均 L2 从 0.299 降至 0.186；ROI 软惩罚改善能量集中但误差回升，450/633 nm 仍未达到设定的 0.85 ROI 功率 | [Step 9 报告](fourier_optics_step9_report.md)、[脚本](fourier_optics_step9_height_optimization.py)、[数据](fourier_optics_step9_results.json) |
| 10. 多起点、ROI maximin 与密集波段 | 三个起点下增广拉格朗日最低 ROI 为 0.811–0.822；只优化三色最差 ROI 的局部解最高约 0.830，但平均 L2 回升到 0.592；400–700 nm 每 5 nm 的 ROI 在 400 nm 降至 0.56–0.63 | [Step 10 报告](fourier_optics_step10_report.md)、[脚本](fourier_optics_step10_robustness.py)、[数据](fourier_optics_step10_results.json) |

## 当前判断

在理想标量近轴模型中，λf 恒定足以让同一个、与波长无关的屏函数在固定物理坐标上产生相同 Fourier 强度。固定高度浮雕会随波长改变相位；直接优化同一浮雕高度可以降低设计波长处的三色图案误差，但误差和 ROI 能量相互制约。多起点优化没有找到三色最低 ROI≥0.85 的解；只优化 ROI 的最佳局部解最低 ROI 约 0.830、平均 L2 约 0.592。对 400–700 nm 的 5 nm 扫描显示三色设计明显不保证带边表现。所有这些是理想标量薄屏模型结果，不是器件性能保证。

## 可复用流程与边界

模块代码已完成，包含 `make_screen`、`make_complex_screen`、单色与多色 Fourier 传播、熔融石英浮雕相位和归一化强度指标；示例验证复场逆 DFT误差为约 \(6.5\times10^{-16}\)，并复跑三类典型目标。接口与变量单位见 [README](fourier_optics_framework_README.md)，完整结论见 [综合报告](fourier_optics_final_synthesis.md)。

现阶段可以收尾：如果继续到真实器件层面，需要先指定材料/层数/透镜拓扑、NA、目标波段、允许图案误差、绝对效率门槛和最小制造尺度，再做结构级 RCWA/FDTD 或实验校准。当前标量代码不包含这些器件条件，也不能从局部优化失败推断严格不可行。
