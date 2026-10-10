# 第二阶段：RGB 多阶衍射共定位

这一阶段根据多波长衍射论文中的核心思路，使用不同波长的不同衍射阶次来补偿色散：

```text
m_B * lambda_B ~= m_G * lambda_G ~= m_R * lambda_R
```

本项目采用：

```text
lambda_B = 450 nm, m_B = 7
lambda_G = 532 nm, m_G = 6
lambda_R = 638 nm, m_R = 5
```

对应的乘积为：

```text
3150 nm, 3192 nm, 3190 nm
```

相对误差约为 1.3%。因此三色的第 7、6、5 阶会在相近的角度出现。

## 运行

在项目根目录执行：

```powershell
python .\phase2_multiorder\simulate_multiorder.py
```

结果输出到：

```text
phase2_multiorder/results/
```

主要输出：

- `phase2_angular_intensity.png`：普通同阶光栅与多阶补偿光栅的角谱比较
- `phase2_screen_coordinate.png`：把角度转换成焦距 100 mm 的屏幕坐标
- `phase2_peak_positions.png`：三色峰位和目标共定位位置
- `phase2_order_search.png`：整数衍射阶次搜索结果
- `phase2_metrics.csv`：峰位、角度误差、屏幕误差和归一化效率
- `phase2_summary.json`：参数与主要结论

## 物理模型

有限周期多缝光栅的角谱强度近似为：

```text
I(theta, lambda) =
  sinc^2(pi*a*sin(theta)/lambda)
  * [sin(N*pi*d*sin(theta)/lambda)
     / (N*sin(pi*d*sin(theta)/lambda))]^2
```

其中：

- `a` 是单缝宽度；
- `d` 是光栅周期；
- `N` 是周期数；
- `theta` 是衍射角。

光栅主峰满足：

```text
d*sin(theta_m) = m*lambda
```

## 重要实验限制

多阶方案首先补偿的是**角度位置**，不是任意二维图案的全部形状。实际实验中需要：

1. 在光栅后用空间滤波器选取蓝色第 7 阶、绿色第 6 阶、红色第 5 阶；
2. 或使用窄带角度选择器分别记录三个颜色；
3. 再比较三色峰位、峰宽、衍射效率和串扰。

如果不进行阶次选择，所有颜色的其他衍射阶也会同时存在。

## 与论文的关系

- Aieta 等人的工作通过波长相关相位梯度直接让多波长获得相同偏转角。
- Wang 等人的工作通过波长相关的幅度和相位调制，在同一观察面重构多波长图案。
- 本阶段采用更适合普通光栅实验的离散阶次补偿，作为从单缝走向多波长全息的中间验证。
