# 第一阶段：RGB 单缝、双缝和多缝衍射基线

这一阶段只回答一个基础问题：

> 普通狭缝或普通多缝，能否让 450 nm、532 nm、638 nm 的 RGB 光在同一个物理屏幕坐标上形成相同图案？

答案预期为：不能。图案在归一化坐标 `X/lambda` 下近似重合，但在真实屏幕坐标 `X` 下会随波长缩放。

## 运行

在项目根目录执行：

```powershell
python .\phase1_baseline\simulate_phase1.py
```

输出位于：

```text
phase1_baseline/results/
```

输出文件：

- `single_slit_physical_coordinate.png`
- `single_slit_normalized_coordinate.png`
- `double_slit_physical_coordinate.png`
- `eight_slit_physical_coordinate.png`
- `phase1_metrics.csv`
- `phase1_summary.json`

## 模型参数

- 蓝光：450 nm
- 绿光：532 nm
- 红光：638 nm
- 传播距离：0.30 m
- 单缝宽度：100 um
- 双缝/多缝中心间距：400 um
- 多缝数量：8
- 屏幕坐标范围：-8 mm 至 +8 mm

## 理论预期

单缝 Fraunhofer 衍射强度为

```text
I(X, lambda) = sinc^2(pi*a*X/(lambda*z))
```

第一暗纹位置为

```text
X1 = lambda*z/a
```

因此在本参数下，第一暗纹位置约为：

```text
B: 1.350 mm
G: 1.596 mm
R: 1.914 mm
```

红色主瓣相对蓝色主瓣的尺寸比约为 `638/450 = 1.418`。

## 第一阶段应完成的实验

1. 分别用 450 nm、532 nm、638 nm 光照明单缝。
2. 在相同传播距离和相同相机倍率下记录三张强度图。
3. 通过暗纹位置测量波长缩放关系。
4. 用双缝和 8 缝样品观察干涉条纹和主峰位置的颜色变化。
5. 将实验曲线与 `phase1_metrics.csv` 和四张理论图比较。

## 实验建议

- 先逐色测量，不要一开始同时打开 RGB。
- 相机曝光、增益和镜头位置保持不变；若动态范围不同，记录原始图后再归一化。
- 使用窄带滤光片分别隔离 R、G、B。
- 用 100 um 左右的单缝开始；缝太宽会使主瓣过窄，缝太窄会使图案超出相机范围。
- 观察距离 0.30 m 是第一版仿真参数，不要求实验必须完全相同；若使用透镜后焦面，则改用 Fraunhofer 角度坐标比较。

## 与后续阶段的关系

这一阶段不是要解决 RGB 同图案，而是建立不可绕开的对照组。第二阶段将引入多阶光栅，尝试让

```text
m_B * lambda_B ~= m_G * lambda_G ~= m_R * lambda_R
```

从而先实现 RGB 亮点、条纹或少数峰的共定位。第三阶段再使用计算全息或波长相关相位器件推广到二维图案。
