# Fourier 光学研究代码：可复用接口

实现文件：[`fourier_optics_framework.py`](fourier_optics_framework.py)。使用 NumPy 与 SciPy，不依赖 PyOptica。数组采用偶数边长、方形、居中网格；所有长度参数用米，波长也用米。

## 常用流程

```python
import numpy as np
from fourier_optics_framework import make_screen, simulate_multicolor

# target_intensity 是定义在期望 Fourier 面网格上的非负强度；
# pupil_amplitude 是输入屏面场振幅，圆孔内为 1、外为 0。
design = make_screen(target_intensity, input_amplitude=pupil_amplitude,
                     iterations=300, seed=1)
screen = design["screen"]

outputs = simulate_multicolor(
    screen,
    wavelengths_m=np.array([450e-9, 532e-9, 633e-9]),
    input_pitch_m=5e-6,
    reference_wavelength_m=532e-9,
    reference_focal_length_m=0.2,
    system="diffractive",
)
```

`make_screen` 用 Gerchberg–Saxton 迭代只调输入屏相位。输出强度目标不含相位；算法会迭代寻找相容的 Fourier 面相位。返回字典含复透射率 `screen`、相位、误差历史与归一化目标。算法非凸，种子、初始相位、孔径和迭代轮数都会影响结果，因此“任意强度目标”表示可以作为目标输入，不承诺精确实现。

`make_complex_screen(target_field, input_field, normalize_to_passive=False)` 适合离散 DFT 完全匹配的复场目标。它用逆 DFT 直接求出所需的屏面场，再除以输入场。这个变换在离散线性模型中精确；但若透射振幅超过 1、输入场有零点而目标要求非零场，或屏幕可用区域不足，就不能由被动相位屏无损实现。可选的被动归一化会统一缩放目标场，并返回 `target_scale`。

## 传播和坐标

`simulate` 返回 Fourier 焦平面的复场、强度、归一化强度与物理坐标。输入间隔为 Δx、数组边长为 N 时：

\[
\Delta f=\frac{1}{N\Delta x},\qquad
\Delta X=\lambda f\,\Delta f=\frac{\lambda f}{N\Delta x}.
\]

普通 Fourier 系统保持焦距 `f` 固定，频率分量对应的焦平面坐标按 `λ f` 缩放；`system="diffractive"` 则人为采用 `f(λ)=f_ref λ_ref/λ`，使 `λf` 恒定、各波长落在同一物理网格。此选项是理想数学算子设定，不代表单片熔融石英浮雕能自动实现该焦距色散。普通模式额外返回重采样到参考网格的 `common_normalized_intensity`；该量适合比较图样，但会受到插值误差。

屏幕空间采样的 Nyquist 频率为 `1/(2 Δx)`。系统孔径/数值孔径还会进一步截断频谱；超出系统 NA 的目标空间频率不可达。有限输入孔径也会使 Fourier 面目标产生有限宽度，不能生成理想点状强度。

## 色散浮雕

`fused_silica_index(λ)` 使用 Malitson 20 °C 熔融石英 Sellmeier 拟合；`relief_screen(height_m, λ, pupil)` 返回

\[
t_\lambda(x,y)=P(x,y)\exp\!\left[i\,2\pi\frac{n(\lambda)-1}{\lambda}h(x,y)\right].
\]

高度 `h` 固定时，相位会随波长变化。示例 [`fourier_optics_framework_example.py`](fourier_optics_framework_example.py) 从任意三光斑目标重新生成一个屏，并读取既有三光斑、圆环、字母 A 设计，对照理想无色散屏、固定焦距与熔融石英主值浮雕的三波长输出。

## 指标与模型边界

`intensity_metrics` 报告归一化强度的相对 L2、平方 Bhattacharyya fidelity 和可选 ROI 功率比例。归一化 ROI 功率不是总透过率或实际衍射级次效率。焦平面场采用标量近轴薄透镜 Fourier 变换；没有包含反射/吸收、偏振、矢量电磁效应、基底界面、制造公差、灰阶量化、最小特征尺寸或实际 DOE 的级次效率。

熔融石英色散式的适用波长范围来自原始测量论文 0.21–3.71 μm。本研究的 400–700 nm 波段位于其范围内。[Malitson, JOSA (1965)](https://opg.optica.org/josa/abstract.cfm?URI=josa-55-10-1205)。当浮雕局部特征接近波长时，应使用 RCWA/FDTD 等电磁方法评估效率，而不应把标量传播结果当成器件效率。[DOE 标量与电磁效率比较](https://www.sciencedirect.com/science/article/abs/pii/S0030401814009535)。
