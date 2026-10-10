# 多波长衍射光学文献整理

当前目录用于保存论文 PDF、补充材料和阅读笔记。由于当前运行环境对部分出版社站点的 PDF 下载请求返回网络拒绝，论文先记录 DOI、公开入口和与本项目的关系；下载成功后将 PDF 放入本目录并按下列文件名保存。

## 优先阅读

### 1. Kim, Domínguez-Caballero & Menon (2012)

- 题目：*Design and analysis of multi-wavelength diffractive optics*
- 期刊：Optics Express 20(3), 2814–2823
- DOI：`10.1364/OE.20.002814`
- 建议文件名：`Kim_2012_multi_wavelength_diffractive_optics.pdf`
- 作用：最接近本项目的数值方法。作者把实际 DOE 高度作为变量，用扩展的 direct-binary-search 方法设计多波长 Fresnel DOE，并讨论高度量化、波长数量、传播距离和制造误差。

### 2. Chen, Wang, Gu & Tan (2016)

- 题目：*Diffractive optical element with same diffraction pattern for multicolor light-emitting diodes*
- 期刊：Applied Optics 55(1), 159–165
- DOI：`10.1364/AO.55.000159`
- 建议文件名：`Chen_2016_same_diffraction_pattern_multicolor_leds.pdf`
- 作用：与“不同颜色得到相同衍射图案”最直接相关，适合用于项目动机和结果比较。

### 3. Bowman et al. (2015)

- 题目：*Multi-wavelength holography with a single spatial light modulator for ultracold atom experiments*
- 期刊：Optics Express
- 作用：多波长可编程全息的参考。它使用 SLM，不是静态 DOE，因此作为算法和系统背景，不作为本项目的器件模型。

## 阅读顺序

1. 先看 Kim 2012 的问题定义、DOE 高度相位关系和优化思路。
2. 再看 Chen 2016 的“相同衍射图案”指标与光学系统背景。
3. 最后看 Bowman 2015，理解 SLM 与静态 DOE 的差别。

## 本项目采用的物理模型

本项目先不做纳米结构全波仿真，而采用标量角谱传播：

\[
\phi_\lambda(x,y)=\frac{2\pi[n(\lambda)-1]}{\lambda}h(x,y),
\]

其中所有波长共用同一个物理高度 `h(x,y)`。本项目采用 Kim 等（2012）中 Shipley S1813 光刻胶的可见光折射率：`n(532 nm)=1.6482`、`n(633 nm)=1.6347`。传播介质取空气，因而相位差中的背景折射率取 `n_air≈1`。这比给每个波长单独设置相位图更符合静态 DOE 的物理约束。
