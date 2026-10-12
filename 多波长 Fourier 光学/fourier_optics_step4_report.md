# Step 4：固定屏的三波长 Fourier 图案比较

## 本步结论

对同一固定屏，在普通固定焦距透镜下，Fourier 平面中的图案尺寸随波长成比例变化。若理想系统的有效焦距满足 \(f(\lambda)=f_0\lambda_0/\lambda\)，则乘积 \(\lambda f(\lambda)\) 不变，因而 Fourier 变换的物理坐标映射不变。本次数值结果验证了这个理想标量模型；它没有证明一块真实器件能在目标波段内无损、无旁级次地实现该算子。

## 数学模型与推导

设 \(t(x,y)\) 是屏的复透射率，二维 Fourier 变换定义为

\[
T(\xi,\eta)=\iint t(x,y)e^{-i2\pi(\xi x+\eta y)}\,dx\,dy.
\]

在标量、近轴、理想薄透镜模型中，屏位于透镜前焦面、观察面位于后焦面时，后焦面复振幅为

\[
U_\lambda(X,Y)=\frac{e^{ikf}}{i\lambda f}
e^{i\pi(X^2+Y^2)/(\lambda f)}
T\!\left(\frac{X}{\lambda f},\frac{Y}{\lambda f}\right),
\qquad k=2\pi/\lambda.
\]

薄透镜的二次相位透射率为

\[
t_{L,\lambda}(x,y)=\exp\!\left[-i\frac{\pi(x^2+y^2)}{\lambda f(\lambda)}\right].
\]

令参考波长为 \(\lambda_0\)，参考焦距为 \(f_0\)，并令

\[
f(\lambda)=f_0\frac{\lambda_0}{\lambda},\qquad
\lambda f(\lambda)=\lambda_0 f_0.
\]

那么透镜二次相位的系数 \(1/[\lambda f(\lambda)]\) 不随波长变化；变换坐标 \((X/(\lambda f),Y/(\lambda f))\) 也不随波长变化。前置因子中的 \(1/(\lambda f)\) 同样不变，剩下的整体传播相位及后焦面二次相位只改变复场相位，不改变强度。因此在这组理想假设下，同一个波长无关的 \(t(x,y)\) 会给出相同的物理强度图案。

## 数值实验

沿用 Step 3 的三光斑相位屏（最佳起点 260930），不随波长重算屏函数。取 \(\lambda=450,532,633\) nm，参考值 \(\lambda_0=532\) nm、\(f_0=0.2\) m。比较两种模型：

1. **普通透镜：** \(f(\lambda)=f_0\)。物理图案坐标按 \(X_\lambda/X_{532}=\lambda/532\,\mathrm{nm}\) 缩放。
2. **理想补偿：** \(f(\lambda)=f_0\lambda_0/\lambda\)，使 \(\lambda f\) 恒定。

输入网格为 1024×1024、间距 5 µm、宽 5.12 mm，频率 Nyquist 上限为 100,000 m⁻¹。用 2 倍零填充加密 Fourier 频率采样，并把各波长场插值到同一个 532 nm 参考物理坐标网格比较；该共同网格像素间距为 20.78125 µm。普通透镜自然焦平面像素间距依次为 17.578、20.781、24.727 µm。比较使用总能量归一化的强度分布。

## 结果

相对 L2 误差、相关和 fidelity 都是与同一张 532 nm 目标强度图比较。屏本身是相位型 GS 近似解，因此补偿后各波长相同，但目标残差不会变成零。

| 透镜模型 | 波长 | 有效焦距 | 图案坐标倍率 | 相对 L2 误差 | 全图相关 | 平方 fidelity |
|---|---:|---:|---:|---:|---:|---:|
| 普通固定焦距 | 450 nm | 0.2000 m | 0.8459 | 0.8521 | 0.6854 | 0.6610 |
| 普通固定焦距 | 532 nm | 0.2000 m | 1.0000 | 0.2509 | 0.9681 | 0.8986 |
| 普通固定焦距 | 633 nm | 0.2000 m | 1.1898 | 0.7670 | 0.6574 | 0.6118 |
| 理想逆波长焦距 | 450 nm | 0.2364 m | 1.0000 | 0.2509 | 0.9681 | 0.8986 |
| 理想逆波长焦距 | 532 nm | 0.2000 m | 1.0000 | 0.2509 | 0.9681 | 0.8986 |
| 理想逆波长焦距 | 633 nm | 0.1681 m | 1.0000 | 0.2509 | 0.9681 | 0.8986 |

普通透镜下，右侧离轴光斑的实测峰值位置约为 0.623、0.748、0.894 mm；以 532 nm 的实测峰值为基准按 \(\lambda\) 比例预测，得到约 0.633、0.748、0.891 mm。偏差约为一个至半个输出像素量级，符合离散采样和插值误差。理想补偿下，三种波长的峰值位置均为约 0.748 mm。直接 Fourier 坐标缩放关系与独立强度插值的相对 L2 差在普通模型下约为 \(4.3\times10^{-5}\)，参考波长/理想补偿下接近机器精度。

## 对真实器件的解释边界

- **理想薄透镜算子：** 设定 \(f(\lambda)\) 后，Fourier 坐标关系严格由 \(\lambda f(\lambda)\) 决定。这里的补偿结论是在模型内成立。
- **折射透镜：** 焦距通常由几何形状与材料折射率共同决定，材料色散会改变焦距；一般不自动遵循 \(1/\lambda\) 规律。
- **衍射透镜/DOE：** 常见衍射聚焦具有随频率增大的焦距趋势，即近似 \(f\propto1/\lambda\)，与本步所需的有效焦距趋势相符。但真实 DOE 的相位深度、材料色散、有限相位量化和各衍射级次效率会随波长变化；因此单一理想 Fourier 算子不足以预测实际透射率与效率。
- **超表面：** 可通过单元结构同时设计相位与色散，已有多波长及连续波段消色差设计研究；这需要针对材料、偏振、数值孔径和目标带宽的器件级优化，不能由焦距公式单独保证。

本步使用的复屏在三个波长下被假定完全相同，透镜也被直接建模为理想 Fourier 变换算子。尚未加入实际 DOE/超表面单元、衍射效率、材料色散、偏振响应、相位灰阶、透镜孔径截断或多级次传播。仅验证 3 个离散波长，未验证连续波段。

## 已验证与未验证

**VERIFIED：** 固定 \(f\) 时图案坐标按 \(\lambda\) 缩放；在给定理想算子 \(f(\lambda)=f_0\lambda_0/\lambda\) 时，三种波长的归一化图案数值一致到插值误差量级；普通模型的实测光斑位移符合预测。

**NOT YET VERIFIED：** 实际器件能否达到该焦距色散、各波长衍射效率是否足够、连续带宽下是否保持同一目标，以及多个目标/任意复场下是否存在满足被动性与有限 NA 的共同屏函数。

## 参考资料

1. MIT OpenCourseWare, *MIT 2.71 Optics, Spring 2009, Lecture 18*：薄透镜的二次相位以及后焦平面 Fourier 关系。[课程讲义 PDF](https://ocw.mit.edu/courses/2-71-optics-spring-2009/dc248d321a4a889ee1f863f4e6891397_u6GbFCWIH_0.pdf)
2. Chen et al., *Broadband achromatic dielectric metalenses*, *Light: Science & Applications* (2018)：讨论普通衍射色散及通过单元相位色散实现宽带消色差的设计条件。[论文](https://www.nature.com/articles/s41377-018-0078-x)
3. de Wit et al., *Diffraction and dispersion effects on the wavelength dependence of the focal length of metalenses*, *Applied Optics* 64, 3598–3604 (2025)：分析衍射与材料色散对真实金属透镜焦距色散的共同作用。[Optica Publishing Group](https://opg.optica.org/abstract.cfm?URI=ao-64-13-3598)

## 产物

- `fourier_optics_step4.py`：同一相位屏的三波长传播、坐标插值、误差指标与图表。
- `fourier_optics_step4_results.json`：六种模型/波长组合的焦距、采样间距、误差和光斑位置。
- `fourier_optics_step4_wavelength_patterns.png`：普通透镜与理想补偿下的图案。
- `fourier_optics_step4_wavelength_differences.png`：相对共同目标的有符号强度差。
- `fourier_optics_step4_scaling_metrics.png`：目标误差和离轴光斑位移。

