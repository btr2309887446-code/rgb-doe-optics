# Step 5：复场基底与有限孔径的 Fourier 模式

## 本步结论

有限离散网格上的复振幅场构成 \(\mathbb C^{N^2}\)，二维 DFT 给出一组完备正交基。这个结论针对**复场**，不意味着强度图可以按同一线性方式相加，也不意味着任意复场都能由相位型屏产生。有限圆孔径会截断 Fourier 基函数：焦平面中的理想单像素点变成 Airy 型光斑，不同频率模式也不再正交。

## 定义与命题

给定 \(N\times N\) 的采样复场 \(U[j,k]\)，其空间为

\[
\mathcal H_N=\mathbb C^{N^2},\qquad
\langle U,V\rangle=\sum_{j,k}U[j,k]^*V[j,k].
\]

令二维 Fourier 基函数为

\[
b_{mn}[j,k]=\frac{1}{N}
\exp\!\left[2\pi i\left(\frac{mj+nk}{N}\right)\right],
\quad m,n\in\{0,\ldots,N-1\}.
\]

**命題（有限 DFT 基）**：这些 \(N^2\) 个向量在上述内积下正交归一，因此构成 \(\mathbb C^{N^2}\) 的一组完备基。每个复场都可写为 \(U=\sum_{m,n}c_{mn}b_{mn}\)，并满足 Parseval 等式 \(\|U\|_2^2=\sum_{m,n}|c_{mn}|^2\)。证明直接来自有限几何级数：不同整数频率的指数和为零；基向量含 \(1/N\) 归一化后范数为 1。

连续 Fourier 级数在有限矩形周期域上给出对应结果；完备性是平方可积意义下的收敛。有限圆孔径的物理屏模式为

\[
b^{(P)}_{mn}(x,y)=P(x,y)e^{2\pi i(mx/L_x+ny/L_y)}.
\]

它们的内积等于孔径函数 \(P\) 的 Fourier 变换在两模式频率差处的取值。矩形完整周期网格可保持 DFT 正交；圆孔径通常不能，所以这些截断模式一般不是彼此正交的。

## 为什么“图案”应先指复场

若复场展开为 \(U=\sum_qc_qb_q\)，则强度是

\[
I=|U|^2=\sum_q|c_q|^2|b_q|^2+
\sum_{q\ne r}c_qc_r^*b_qb_r^*.
\]

第二项是相干交叉项。因此强度图案不是复场基系数的线性叠加；谈“基本图案”时应先规定对象是复振幅、非相干强度，还是含交叉项的相干传播。

## 基底选择比较

| 基底 | 完备/正交条件 | 适用优势 | 实际限制与波长因素 |
|---|---|---|---|
| Fourier / DFT | 周期矩形域；连续时在有限周期域的 \(L^2\) 意义下完备 | 频率、衍射级次和角谱直接对应；FFT 计算高效 | 孔径截断产生 sinc/Airy 展宽和模式耦合；焦面尺寸含 \(\lambda f\) |
| Cartesian 像素基 | 有限采样空间的标准单位向量，严格完备正交 | 空间局部、便于描述 SLM 像素和稀疏局部目标 | 单像素传播为 sinc 型扩展；填充因子、像素串扰和有限相位灰阶改变响应 |
| sinc 基 | 理想带限函数空间、Nyquist 采样条件成立时 | 给出带限连续场的采样重建 | sinc 有无限支撑；有限器件截断后不再精确正交 |
| Hermite–Gaussian（HG） | 给定束腰尺度的傍轴横模空间 | 笛卡尔对称、Gaussian 光束和模式耦合 | 有限孔径截断破坏正交性；束腰/传播尺度随波长和系统设置变化 |
| Laguerre–Gaussian（LG） | 给定束腰尺度的傍轴圆柱横模空间 | 圆对称、涡旋与轨道角动量问题 | 有限孔径截断、偏心和像差引入串扰；尺度同样受色散影响 |
| Zernike | 均匀权重单位圆盘上的波前/孔径函数空间 | 圆形瞳面像差、相位误差的紧凑描述 | 不等同于一般像面强度基；中心遮挡、非均匀权重会破坏标准正交性 |

所以 Fourier 基不是唯一基底：不同完备基由不同内积、边界与器件目标定义。周期目标/频谱分析适合 Fourier；局部元件适合像素；Gaussian 传播适合 HG/LG；圆形瞳面波前误差适合 Zernike。稀疏性取决于目标和基底，不能对所有图案指定同一个最稀疏基。

## 数值验证

演示使用 256×256 网格、10 µm 间距、2.56 mm 方形窗口和直径 1.28 mm 的圆瞳；Fourier 平面用 4 倍零填充细采样。对固定随机复场，单位化 DFT 的相对重建误差为 \(3.82\times10^{-16}\)，Parseval 相对误差为 \(3.34\times10^{-16}\)。

在 81 个低阶频率模式 \(m,n\in[-4,4]\) 上，完整矩形网格的最大非对角归一化内积为 \(1.51\times10^{-7}\)（数值精度量级）；同一组模式乘上圆孔径后，最大非对角内积升至 0.722，最近邻模式平均相干度也约为 0.722。此处的相干度是模式内积绝对值，数值越大表示越难把相邻模式视为独立“基本斑点”。

圆孔径内的 \((m,n)=(3,-2)\) 相位斜坡，其理论焦平面峰位为 \((0.12469,-0.08313)\) mm，数值峰位完全落在对应 Fourier 像素上。圆孔径 Airy 第一零点理论偏移 0.10141 mm，数值第一极小值为 0.10391 mm，偏差约 0.24 个细采样像素；屏面/频域 Parseval 相对误差为 \(4.25\times10^{-16}\)。这验证了“有限孔径平面波模式 → 平移后的孔径 Fourier 变换”，而不是理想 Dirac 点。

## 实际屏函数的约束

代数上，任意离散复屏可写作 DFT 指数函数的线性组合；但若要屏满足被动振幅约束 \(|t|\le1\)，或相位型约束 \(|t|=1\)，一般需要对系数施加非线性约束。Fourier 展开本身不构造一个物理上可做的相位型屏。GS 等算法处理的是这种受限逆问题，不是基底完备性的推论。

**VERIFIED：** 离散 Fourier 基完整正交、单位化变换满足 Parseval；圆孔径截断后模式重叠显著；单一截断 Fourier 模式产生与圆孔径 Airy 解析式吻合的有限宽光斑。

**NOT YET VERIFIED：** 没有计算 HG/LG/Zernike 对一批目标图案的实测稀疏度或条件数；没有加入真实像素填充因子、NA 截断、相位量化与跨波长器件响应。不同基底的表格是数学适用域比较，不是某类器件性能的实验排名。

## 参考资料

1. MIT OpenCourseWare, *MIT 2.71 Optics, Spring 2009, Lecture 18*：有限孔径光栅频谱是多个平移 sinc 包络的叠加。[讲义 PDF](https://ocw.mit.edu/courses/2-71-optics-spring-2009/dc248d321a4a889ee1f863f4e6891397_u6GbFCWIH_0.pdf)
2. University of Arizona, *Gaussian Beams and Optical Cavities*：Hermite–Gaussian 解及其在傍轴模式空间中的完备性。[课程讲义 PDF](https://wp.optics.arizona.edu/opti511l/wp-content/uploads/sites/36/2016/04/HeNe_Fall2015_Part_1_511L.pdf)
3. Noll, R. J. “Zernike polynomials and atmospheric turbulence.” *Journal of the Optical Society of America* 66(3), 207–211 (1976). [Optica Publishing Group](https://opg.optica.org/josa/abstract.cfm?uri=josa-66-3-207)
4. Teixeira, R. C. “Maximum coupling efficiency to an optical resonator based on the Laguerre–Gauss decomposition of a beam.” *Optics Express* 28(7), 9541–9551 (2020). [Optica Publishing Group](https://doi.org/10.1364/OE.390126)

## 产物

- `fourier_optics_step5.py`：离散完备性、Parseval、孔径 Gram 矩阵和圆孔径 Airy 模式仿真。
- `fourier_optics_step5_results.json`：误差与有限孔径模式相关度。
- `fourier_optics_step5_basis_and_finite_aperture.png`：Gram 矩阵、二维光斑与解析/数值剖面对照。
