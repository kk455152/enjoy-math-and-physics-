# 动力学可调用的电磁力接口

> **本轮新增完整空间瞬态入口：**请先读 [完整瞬态接口调用说明](完整瞬态接口调用说明.md)，使用 `ComsolTransientStepper` 和同时传入力/力矩的 `predictor_corrector_full_step`。模型原生求解 J 与 JA 磁化历史，短轨迹初始化与反馈试算已成功；跨重网格长轨迹和精度验收仍未通过。下面保留旧接口记录。

> 2026-10-30：最新静磁接口为 `连续磁力地图/shared_parametric_map.py` 的 `SharedStaticForceMap`，采用同一 `(rho-R,N*delta/pi,g,D)` 周期插值方法，分材料/磁环保存八组系数，覆盖 R±15 mm 的整周运动环带，g 和 D 连续输入。用法见 `连续磁力地图/统一参数化方法与调用.md`，示例见 `example_shared_parameters.py`。下文旧单切片接口为历史记录。

> 2026-10-29 更正：下文是旧单切片降阶接口记录，不是此次用户要求的完整瞬态交付。其静磁数据来自待核查的旧几何，禁止用于新的动力学结果。新的静磁批次位于 `..\连续磁力地图`；完整目标与尚未完成的部分见 `完整耦合目标与当前状态.md`。

本目录提供一份可运行的降阶耦合接口。动力学程序给入球心位置、速度、球自转和转盘角速度；接口按每个时刻返回静磁力、低频涡流阻力近似、磁滞损耗阻力估计及它们的合成磁力。它不读取实验最终速度，也不求解球的运动。

当前随包磁力数据只覆盖一条已经求解的切片：GCr15 淬硬态，`N=10, R=80 mm, rho=85 mm, g=6 mm, D=6 mm, 平板厚度=1 mm`，球心位于磁铁投影内，局部切向坐标 `vRel=-5..5 mm`。数据以 41 个 COMSOL 点建立 201 点 PCHIP 密集表；验证报告中的切向力插值误差为力变化范围的 0.382%。这只是本切片的插值误差，不是总模型精度。

材料参数分支已写入代码，但当前随包只有 GCr15 淬硬态有匹配的连续静磁力图和球内场图。H62、PC40、GCr15 退火态要等对应几何切片的两张地图补齐后才能调用；传入没有数据的材料/工况会报错。

## 运行

依赖仅为 Python 3 标准库。把动力学同学输出的轨迹写成 CSV，至少包含：

```csv
t_s,x_mm,y_mm,vx_m_s,vy_m_s,Omega_rot_rad_s,omega_ball_z_rad_s
```

`x_mm,y_mm` 是球心在实验室坐标中的位置；速度用 m/s；`Omega_rot_rad_s` 和球体自转角速度用 rad/s。可额外提供 `omega_ball_x_rad_s,omega_ball_y_rad_s`，但当前磁场截面不能解析非 z 方向自转的涡流力矩。

用当前示例运行：

```powershell
py -3 .\em_force_model.py .\trajectory_example.csv `
  --output .\example_force_output.csv `
  --material GCr15_hardened --N 10 --R-mm 80 --rho-mm 85 --g-mm 6 --D-mm 6
```

也可以把动力学每一步的状态直接交给 Python 接口，见 `example_member_dynamics.py`：

```python
model = ElectromagneticForceModel(force_map, field_map, material="GCr15_hardened",
                                  N=10, R_mm=80, rho_mm=85, g_mm=6, D_mm=6)
rows = model.evaluate_trajectory(states, rotor_phase0_rad=0.0)
Fmag = [rows[-1]["Fmagnetic_x_N"], rows[-1]["Fmagnetic_y_N"], rows[-1]["Fmagnetic_z_N"]]
```

若动力学程序逐步调用，在固定步长积分循环中用 `model.evaluate(state, rotor_angle_rad)`；转盘相位由动力学端按 `Omega_rot_rad_s` 积分并传入。若使用带回退/重复求值的自适应 ODE 求解器，`tau_mag_s>0` 的内部滞后状态不应放在 RHS 副作用里，需把这项状态显式并入 ODE 状态或先固定轨迹再调用 `evaluate_trajectory`。

输出 CSV 的每一行与输入状态一一对应。`Fmagnetic_x/y/z_N` 是 `Fstatic + Feddy + Fhyst`，单位 N。动力学端再加入质量、重力、平板支持力、摩擦、接触约束等。`eddy_torque_Nm` 明确标为 `not_computed`，不能当成零涡流力矩使用。

变速转盘通过每行 `Omega_rot_rad_s` 输入，接口对相邻时刻用梯形积分累计转盘相位。`--rotor-phase0-rad` 指定第一时刻转盘初相位，默认 0，对应第 0 块磁铁中心在初始 `+y` 方向。时间必须严格递增。

## 力模型

静磁基线直接读取 COMSOL 的 `Fr_orbit_N,Ftheta_orbit_N,Fz_N`，在同一材料和几何切片内按 `vRel_mm` 线性插值，再旋转至实验室坐标。

低频涡流项是球体低磁雷诺数的降阶估计，不是把现有固定球心时域算例误当作任意轨迹：

\[
C_e=\frac{2\pi}{15}\sigma a^5,\qquad
F_{e,\theta}\simeq-C_e v_{rel,\theta}
\left\langle\sum_{i=x,y,z}\left(\frac{\partial B_i}{\partial s}\right)^2\right\rangle_V.
\]

其中 `a=D/2`，`vrel_theta=rho*(theta_dot-Omega_rot)`，`s` 是相对磁场沿球心轨道的弧长。空间导数从 COMSOL 球内竖直中心线 `B(z,vRel)` 做有限差分；体积平均用球截面权重 `1-eta^2` 近似。该式保证涡流阻力功率不为正，但省略了磁场全体积非均匀性、强趋肤效应以及球自转造成的涡流力矩。

磁滞项是等效能量损耗阻力。GCr15 的可逆骨干曲线沿用参数表的 Fröhlich–Kennelly 形式；工作场幅值从球内 `H` 场样本估计，损耗密度采用

\[
w_h\simeq4H_c\min\left[B_r,B_{anh}(H_{amp})\right],\quad
P_h=w_hVf,\quad f\simeq\frac{N|\omega_{field}|}{4\pi}.
\]

`B_anh(H)` 是参数表给出的可逆骨干曲线：GCr15 淬硬/退火分别用 `2H/(H+53 kA/m)` 和 `2H/(H+16 kA/m)`；PC40 用工作簿拟合 `H/(346+1.67H)`。GCr15 淬硬态用 `Hc=2.9 kA/m, Br=0.7 T`，敏感性范围 `Hc=2.03..3.77 kA/m, Br=0.5..0.9 T`；这些不是本钢球实测回线。退火态用表格工程估算 `Hc=0.4 kA/m, Br=1.35 T`，范围分别 `0.2..0.6 kA/m` 和 `1.2..1.5 T`。PC40 用资料表 `Hc=14.3 A/m`，并从其两点拟合 B–H 曲线在 `H=Hc` 处估算 `Br`；PC40 的 `Hc` ±30% 只作为本接口的灵敏度范围，不是厂家公差。H62 按非磁性材料不加入铁磁磁滞项。

磁滞损耗按广义相对运动在平移和绕 z 自转之间分配，保证

\[
\mathbf F_h\cdot\mathbf v_{rel}+T_{h,z}\omega_{rel,z}=-P_h.
\]

这个等效阻力用于动力学敏感性分析，不是实测磁滞回线，也不表示 COMSOL 已求解磁滞磁场。`--hysteresis-scale` 可用于不确定性分析；建议至少比较 `0.5, 1, 2`。`--tau-mag-s` 对等效磁滞力施加一阶滞后，默认 0（关闭）；它不是实际材料的磁化弛豫时间，B–H 曲线本身不能确定它，必须作为敏感性变量或以后用测量标定。

## 使用边界

- 轨迹半径必须等于当前地图的 `rho=85 mm`，`R,g,D,N,材料` 也必须与地图完全一致。其它参数组合需先补 COMSOL 磁力与球内场地图；接口拒绝用邻近参数替代。
- 当前只在磁铁矩形投影内返回力；投影外及数据范围外报错。没有把未计算区当成零力。
- 目前只对该 GCr15 淬硬态切片有随包磁力/场数据。程序可换用其它材料和参数对应的 CSV，但没有对应地图时会报错。
- 时变 COMSOL 模型仍是固定球心、不自转、恒角速度启动瞬态。其 `J×B` 数据不能作为任意轨迹的直接查表值，也没有加到本接口结果中，避免与静磁总力重复计数。
- 涡流近似的 `eddy_skin_parameter` 大于 0.1 时会在 `model_flags` 提示趋肤/扩散效应可能不可忽略；大转速下应重新做移动球 COMSOL 全耦合。
- 仅用一维高度场估计三维场梯度，球自转涡流力矩未算。磁滞力方向及损耗也是能量法近似。模型适合先做动力学结构和敏感性分析，不应当作最终实验拟合结果。
- `tau_mag_s` 是等效滞后参数，不是根据静态 B–H 数据推导出的材料常数；建议先跑 `0, 1e-4, 1e-3, 1e-2 s` 的敏感性组，不能只报告一个未经测量的值。

## 参数与材料来源

GCr15 电导率 `5.3 MS/m`、B–H 骨干曲线 `Bs=2 T`、淬硬/退火 `h0=53/16 kA/m`、矫顽力和剩磁的估算范围来自 `GCr15钢球_磁力动力学仿真参数表(3).xlsx`。淬硬态 Hc 参考 Ankener et al., *Sensors* 22(12), 4428 (2022), DOI: `10.3390/s22124428` 中相近 AISI 52100 淬火态数据；并非本用户钢球的实测值。PC40 参数来自同一工作簿所列 TDK 数据与近似 B–H 拟合。H62 电导率取工作簿推荐值。

本模型输出不包含摩擦和重力。动力学总方程可以写成：

\[
m\ddot{\mathbf x}=\mathbf F_{magnetic}+\mathbf F_{friction}+\mathbf F_{gravity}+\mathbf F_{contact}.
\]

其中 `Fmagnetic` 由本接口给出，摩擦、法向约束及自转接触力矩仍由动力学程序负责。
