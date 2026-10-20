# 固定系统和孔径下的自由屏函数逆向设计

程序：`fourier_optics_step11_free_screen_optimization.py`。依赖 NumPy、SciPy、Matplotlib，无需 PyTorch、GPU 或真实材料折射率。

## 比较的对象

所有模型采用同一张目标强度图、同一孔径、同一入射平面波和固定焦距 0.2 m 的 Fourier 透镜。默认训练波长为 450、532、633 nm，另扫描 400–700 nm。**没有使用随波长改变焦距的补偿透镜。**

1. `phase`：任意逐像素相位，`q=exp(i*phi)`。
2. `complex`：所有波长共用任意逐像素复场，`q=a+i*b`。
3. `spectral_degree_1/2`：`q(lambda)=sum(c_k*T_k(s(lambda)))`，其中 `c_k` 为逐像素复系数，`T_k` 为 Chebyshev 多项式。默认分别比较一阶、二阶。

三类模型均在传播前按 `u=q/||q||` 归一化，并在孔径外设为零。对应透射函数为 `t=sqrt(孔径像素数)*u`，所以在均匀入射下各波长具有相同的离散总透射功率。自由复屏可以局部放大，峰值透射振幅可能超过 1；这是用户指定的数学自由度，不是被动材料。

色散模型是**多项式复场再逐波长归一化**，归一化后的透射函数本身一般不是同阶多项式。梯度包括这个归一化的导数。

对于三个训练波长，二阶基矩阵满秩时可以表达三个独立复屏。低训练误差不等于连续波段的低误差；一阶和二阶结果都要查看训练外扫描。

## 传播、坐标和残差

屏面总宽度继承旧数据，默认 5.12 mm；改变 `--grid` 时保持物理宽度不变。孔径按原存档的物理坐标重采样，报告记录原始及重采样孔径面积。默认观察面为 256×256、15.625 μm 像素，约 4 mm 视场；观察面网格独立于屏面网格。

矩形 chirp-z transform 直接计算同一物理观察坐标上的离散 Fourier 积分，避免把不同波长相同数组索引误当作相同坐标。数学上：

`alpha = dx*dX/(lambda*f)`

`K[k,j] = exp(-2*pi*i*alpha*(j-N/2)*(k-M/2))`

`U = alpha * K @ u @ K.T`，`I = abs(U)**2`。

目标从旧存档的参考物理坐标插值到观察网格；程序检查采样后的目标总量，不允许明显裁掉目标后静默归一化。新程序的强度残差为：

`relative_L2(lambda) = ||I(lambda)-target|| / ||target||`

`loss = mean(relative_L2(lambda)**2)/2`。

**不按观察窗内功率重新归一化模拟图案。** ROI 功率以传播前归一化的总功率为分母。另存一列观察窗重新归一化的误差，仅作诊断，不用于优化。

模型沿用项目的点采样标量 Fourier 近似及离散 Parseval 功率约定，观察窗限制在各波长的原生 Nyquist 范围内。未包含实际像素面积的 sinc 包络、材料、电磁矢量效应、反射或吸收；不能把这个模型的功率比例解读为实际器件效率。新结果的传播系统、观察窗口和损失定义与 Step 9 不同，不能把两步的 L2 数字直接横向比较。

## 运行

在本项目文件夹打开终端：

```powershell
# 只做传播算子、伴随、梯度、归一化和逆变换自检
python fourier_optics_step11_free_screen_optimization.py --check-only

# 三个既有目标，多个初值；新目录必须尚不存在
python fourier_optics_step11_free_screen_optimization.py --starts 2 --iterations 250 --output my_comparison

# 单独优化三光斑；增加预算
python fourier_optics_step11_free_screen_optimization.py --targets three_spots --starts 3 --iterations 1000 --output my_long_run

# 保持观察面不变，将屏面分辨率提高到 512
python fourier_optics_step11_free_screen_optimization.py --targets three_spots --grid 512 --detector-grid 256 --starts 1 --iterations 300 --warm-start step11_comparison/three_spots/designs.npz --output my_refinement

# 增加训练波长，考察宽带目标；可把多项式阶数继续提高
python fourier_optics_step11_free_screen_optimization.py --targets three_spots --train-nm 400,450,500,532,550,600,633,650,700 --degrees 1,2 --starts 2 --iterations 400 --output my_broadband
```

`--targets` 也接受 `annular_ring`、`letter_A` 和逗号分隔列表。`--target-npy path.npy` 接受定义在当前物理观察网格上的非负二维强度数组，大小必须等于 `--detector-grid`。目标只提供强度，不要求给定目标相位。自定义目标仍沿用存档中的固定孔径。

如果使用 `--warm-start`，程序会保留同一模型的旧解，以及较低自由度模型的解。此时可能比 `--starts` 多一个必要候选；不会把旧文件覆盖掉。

## 优化和可信度

优化器为使用解析伴随梯度的 L-BFGS-B（本问题不加参数上下界），包含多初值和最佳迭代保留。更灵活模型包含前一模型，初始化也显式包含前一模型的最佳场，因此保存的训练目标函数不会因模型扩展而变差。训练外误差没有这个保证。

内置验证覆盖：

- 方形与矩形 CZT 相对于显式 Fourier 矩阵的误差；
- 伴随内积恒等式；
- 四种参数化的方向有限差分梯度；
- 复屏总幅度缩放不变性和模型嵌套；
- 完整方孔径、单波长、匹配 DFT 网格的精确反演基准。

每次运行保存独立目录，每完成一类模型就保存检查点。输出包括 `results.json`、`metrics.csv`、`designs.npz`、`report.md`、`patterns.png`、`scan_and_convergence.png`，上级目录另存数学检查和汇总。

`designs.npz` 中的 `*_coefficients` 和 `*_theta` 为 float64/complex128 的优化变量；`*_screens` 为训练波长实际归一化透射函数，`*_images` 为输出强度。要计算新波长，应使用保存的系数、`band_nm`、孔径和同样的归一化规则，不能直接把某个训练波长的透射数组复制过去。

有限迭代的误差只给出当前离散模型最小值的**上界**。需比较更多初值、更长预算和更细屏网格；离散波长扫描也不是整个连续波段的证明。提高屏网格既改善孔径离散，也增加可设计空间自由度，不应将所有误差变化都解释为积分误差。

## 已完成的实验与单模型继续优化

本轮实际结果与结论见 `fourier_optics_step11_report.md`。`step11_comparison` 是三个目标的主比较；`step11_smoke` 只是五步端到端检查，不应作为最终优化结果。

辅助程序 `fourier_optics_step11_validate.py` 读取已完成的一个目标目录，保持该目标和物理观察面不变，只继续优化选定模型：

```powershell
# 保持原网格，继续优化二阶复屏
python fourier_optics_step11_validate.py --source step11_comparison/annular_ring --output another_continuation --iterations 300

# 继承二阶屏，增加训练波长
python fourier_optics_step11_validate.py --source step11_comparison/annular_ring --output another_broadband --train-nm 400,450,500,532,550,600,633,650,700 --iterations 350

# 对字母 A 的二阶复屏进行分辨率检查
python fourier_optics_step11_validate.py --source step11_comparison/letter_A --output another_refinement --grid 512 --iterations 300
```

默认继续优化 `spectral_degree_2`；可通过 `--model phase`、`--model complex` 或 `--model spectral_degree_1` 选择其他已保存模型。该程序使用继承解作为一个起点，属于敏感性检查，不应当作重新进行的公平多初值比较。

## 可见光全波段多尺度目标实验

新增实验入口 `run_visible_multiscale_design.py`，目标生成器为 `make_multiscale_resolution_target.py`。它对 256×256、4 mm 视场的抗锯齿多尺度线条图，在 400–700 nm 每 10 nm、并加入 488/532/633 nm 的 34 个点上联合优化二阶和三阶 Chebyshev 复屏。二阶模型的每个多起点解零扩展到三阶，新增的三阶系数初始为零；两种模型使用相同波长、目标、损失、迭代预算和起点数。

损失为 `0.7 × 平均相对 L2² + 0.3 × 最差 10% 相对 L2² 的经验 CVaR + 0.1 × 平均 max(0, 0.80 − ROI 功率)²`。最差 10% 对有限训练点数使用分数边界样本，精确保持 10% 尾部权重。ROI 是目标值不低于峰值 1% 的像素；其功率以传播前屏面总功率为分母。

400–700 nm 以 1 nm 扫描，逐点检查相对 L2 ≤0.10 且 ROI 功率 ≥0.80。多尺度目标未通过时，三阶模型最多补入两轮、每轮最多六个误差/ROI 约束最差且彼此相隔至少 5 nm 的训练点。原有圆环目标作控制组，不进行自适应补点。380–780 nm 的 1 nm 扫描仅作为范围外压力测试，单独汇报。

屏面默认 320×320，观察面保持 256×256、15.625 μm 间距和 4 mm 视场。320 屏网格用于使 380 nm 压力测试处于离散 Nyquist 范围内；目标像素、孔径物理尺寸和焦距保持原 Step 11 的约定。二阶和三阶参数总数随此采样统一增加，阶数比较仍只改变多项式系数数目。

运行前检查数学和损失梯度：

```powershell
python run_visible_multiscale_design.py --check-only
```

完整实验命令（输出路径必须是新目录）：

```powershell
python run_visible_multiscale_design.py --output step11_visible_multiscale_20261020
```

可用 `--starts`、`--iterations`、`--grid` 和 `--adaptive-rounds` 覆盖默认预算。改变 `--grid` 时需同时满足 380 nm 采样检查。结果目录保存生成目标及物理元数据、配置、逐波长 CSV、二阶/三阶系数、每个 1 nm 波长的强度 `.npy`、光谱曲线、代表性共尺度图、优化收敛日志和综合报告。

这项实验仍是固定孔径下自由复屏的标量、点采样 Fourier 模型。即使双门槛全部通过，也不能据此声称真实材料 DOE 的加工可行性或器件效率；ROI 功率是该数值系统中的收集比例。
