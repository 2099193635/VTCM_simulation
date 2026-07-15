# 动力学逆映射两篇论文重规划（2026-06-30）

## 0. 总判断

当前工作不宜继续拆成 A/B/C 三篇。三篇拆法的优点是叙事干净，但会把证据链切得过碎：A 讲约束有效性，B 讲宽频可观性，C 讲几何-刚度根因辨识。现在代码库的正向仿真能力已经显著增强，尤其是结构病害、道床刚度扰动、几何-刚度联合标签和数据集构建流程已经形成，因此更适合整合成两篇：

1. **论文 1：可解释的车轨耦合正向仿真基准与可观性边界**
   重点回答：车载响应中到底包含哪些可用于逆向辨识的信息？不同病害、波长、速度、刚度扰动在动力响应中是否可观？

2. **论文 2：物理约束神经逆算子的几何-刚度联合辨识**
   重点回答：在论文 1 给出的可观性边界内，如何从运营列车响应同时反演轨道几何不平顺与支撑刚度异常，并服务灾后保通和大修决策？

这比原来的 A/B/C 更稳：第一篇把正向仿真、PCEQ、Fisher/可观性、病害响应规律合成一个强方法论基座；第二篇再把 PCNIO/多速度/联合逆向/UQ 放在一个完整逆向辨识故事里。

## 1. 从文献得到的真实方法缺口

### 1.1 现有文献覆盖了“逆向识别”，但缺少“可观性边界”

Zotero 的“动力学反衍”收藏夹中，轨道不平顺反演主要有 Kalman filter、inverse transfer function、detrending integration、Bayesian deep learning、PINN/PINO/神经算子几类。它们共同的问题不是没有模型，而是大多直接给出一个反演器，却没有回答：

- 哪些波长段可由车体/构架/轮对加速度稳定观测？
- 几何不平顺与支撑刚度扰动是否会产生混淆？
- 单速度、低 SNR、小样本、跨车辆参数时，逆问题什么时候失去可辨识性？
- 物理约束到底改善的是误差，还是改善了逆问题的条件数？

因此缺口应从“又提出一个逆向模型”上移到“建立运营列车逆向辨识的可观性与约束有效性方法论”。

### 1.2 现有神经算子文献强在 PDE inverse，弱在车轨多体动力学 inverse

Neural inverse operators、latent neural operators、PINO/PINN 文献给了方法学语言，但轨道领域仍缺少三件事：

- 将**时域多体动力学响应**映射到**空间轨道状态函数**的条件神经逆算子；
- 将车辆参数、速度、传感器位置等作为条件变量显式纳入逆算子；
- 将物理一致性从普通正则项提升为可解释的 Fisher 信息/条件数改善。

这正好对应 PCNIO 的方法优势。

### 1.3 轨道工程文献强在单病害检测，弱在根因联合辨识

空吊、过渡段沉降、轨道刚度估计、轨道不平顺估计的文献各自存在，但大多将几何异常和结构刚度异常分开处理。实际维护决策需要回答的是：

> 这是几何可恢复病害，还是支撑刚度退化导致的不可恢复病害？

这给第二篇的“几何-刚度联合逆向辨识”提供了更高层的工程问题，而不是只做一个双输出模型。

## 2. 当前项目证据基础

### 2.1 已具备的正向仿真基础

当前 `vtcm-simulation` 已具备：

- 车辆 35 DOF + 双轨模态 + 轨枕-道床离散结构；
- 随机谱/IFFT、外部导入、局部结构病害注入；
- 两点轮轨接触、Hertz 法向、Kalker-SHE 切向力；
- 全步线路几何预计算、多位置插值、曲线等效力；
- `structure_defect_fastener_void_scan`：扣件失效、空吊及组合工况；
- `transition_settlement_scan` 与 `transition_settlement_random_scan`：过渡段沉降；
- `ballast_condition_stiffness_only_100m_scan` 与 `ballast_condition_sync_eta_100m_scan`：100 m 刚度/阻尼扰动；
- `joint_inverse_stiffness_geometry_scan`：同一几何种子下刚度软化/硬化反事实对比；
- `pipeline/build_joint_inverse_dataset.py`：从 `simulation_result.npz` 生成 `obs -> geometry + stiffness` 的窗口化监督数据集。

### 2.2 已有结果的含义

已有结果不是最终论文结果，但足以支撑选题方向：

- 100 m 刚度软化 `eta_k=0.2` 在车体/构架位移 RMS 上有明显响应变化，车体垂向位移 RMS 增幅约 26.9%；
- 刚度硬化 `eta_k=5` 对位移 RMS 呈下降趋势，车体位移 RMS 约 -4.5%；
- 单纯车体加速度 RMS 对刚度异常不够敏感，说明需要多通道、多尺度、可能还要多速度或力响应辅助；
- 已生成 `ballast_random_joint_inverse_dataset.npz`，包含 144 个窗口，输入 10 个响应通道，输出 4 个几何通道和 2 个刚度通道；
- 目前联合逆向数据集更像 proof-of-data-readiness，而不是足以训练高水平模型的正式 benchmark。

## 3. 两篇论文的新定位

## 论文 1

### 暂定标题

**Observability Boundaries of In-Service Railway Track State Inversion: A Vehicle-Track Coupled Simulation Benchmark for Geometry and Support Stiffness Defects**

备选中文定位：

**运营列车轨道状态逆向辨识的可观性边界：面向几何与支撑刚度病害的车轨耦合仿真基准**

### 一句话论点

在运营列车响应用于轨道状态逆向辨识之前，必须先回答“哪些病害状态在车辆动力响应中可观”。本文基于高保真车轨耦合动力学仿真，构建几何不平顺、空吊、过渡段沉降和支撑刚度扰动的反事实基准，并用响应敏感性、频域传递特性和 Fisher 信息刻画不同速度、波长和结构参数下的可观性边界。

### 核心贡献

1. **基准贡献**：构建一个覆盖几何病害与支撑刚度病害的车轨耦合正向仿真基准，提供反事实样本：同一几何输入下仅改变刚度/阻尼/空吊/沉降。
2. **理论贡献**：提出速度-波长-传感器通道的可观性分析框架，用 Fisher 信息或响应雅可比条件数解释长波、刚度和几何耦合下的不可观区域。
3. **工程贡献**：给出运营列车监测中可用于逆向辨识的传感器组合和工况边界，例如车体位移/加速度、构架响应、轮轨力对刚度异常的敏感性差异。
4. **方法论贡献**：把 PCEQ 从“物理约束有效性评估”重定位为“逆向辨识前的可观性-约束有效性诊断框架”。

### 方法论

论文 1 不以神经网络为主角，而以正向系统和可观性为主角。

方法链条：

1. **正向系统建模**：车辆 35 DOF、钢轨模态、轨枕-道床离散、轮轨非线性接触。
2. **病害参数化**：随机不平顺、枕下空吊、过渡段沉降、支撑刚度软化/硬化、刚度-阻尼同步扰动。
3. **反事实设计**：固定几何种子，只改变结构状态；固定结构状态，只改变速度/车辆参数；以此分离几何激励与结构参数激励。
4. **可观性指标**：
   - 响应增量 RMS / peak / PSD change；
   - 速度-波长传递函数；
   - Fisher 信息或局部雅可比奇异值；
   - 几何-刚度混淆度，例如 $\langle \partial a/\partial z, \partial a/\partial k \rangle$。
5. **边界归纳**：输出“哪些状态可反演、哪些需要多速度/多通道/先验约束”的图谱。

### 实验设计

1. **E1 正向模型验证与基准构建**
   - 展示模型结构、输出通道、结果归档、病害注入能力；
   - 使用已有 `耦合动力学计算流程与原理.md` 作为方法骨架。

2. **E2 几何病害响应**
   - 空吊 1/2/5 mm；
   - 过渡段沉降不同幅值/波长；
   - 输出响应通道敏感性图。

3. **E3 支撑刚度响应**
   - `eta_k=0.1/0.2/0.5/1/2/5/10`；
   - 刚度-only vs 刚度-阻尼同步；
   - 输出位移、加速度、轮轨力的敏感性差异。

4. **E4 几何-刚度混淆反事实**
   - 同一几何种子下 baseline、loose、hardened；
   - 证明单一响应特征不足以区分几何和结构状态。

5. **E5 可观性边界**
   - 速度 160/215/250 km/h 起步，后续扩展到 300/350 km/h；
   - 波长 D1/D2 分段；
   - 用 Fisher/雅可比条件数给出可观性图谱。

### 章节结构

1. Introduction：为什么逆向辨识前必须先研究可观性边界。
2. Vehicle-track coupled simulation benchmark：正向模型与病害注入。
3. Observability and counterfactual analysis：可观性指标与反事实设计。
4. Results：几何、空吊、沉降、刚度扰动的响应规律。
5. Discussion：哪些状态适合逆向反演，哪些需要多速度/物理先验/联合辨识。
6. Conclusion：给出论文 2 的方法边界。

### 推荐目标

优先：MSSP / Vehicle System Dynamics。  
理由：这篇更偏动力学、信号、可观性和仿真基准，不要强行投 AI 工程顶刊。

## 论文 2

### 暂定标题

**Physics-Conditioned Neural Inverse Operator for Joint Identification of Track Geometry and Support Stiffness from In-Service Vehicle Responses**

备选中文定位：

**面向运营列车响应的物理条件神经逆算子：轨道几何与支撑刚度联合辨识**

### 一句话论点

仅反演轨道几何不平顺无法支撑灾后保通和大修决策，因为同样的车辆响应异常可能来自几何超限，也可能来自支撑刚度退化。本文提出物理条件神经逆算子，将车辆响应、速度和车辆参数映射为轨道几何与支撑刚度的双输出空间函数，并通过物理一致性、多速度/多通道条件化和不确定性校准提升联合辨识稳定性。

### 核心贡献

1. **方法贡献**：提出 PCNIO-Joint，将原 PCNIO 从单输出几何反演扩展为双输出 $(z_r(x), \eta_k(x))$ 联合逆算子。
2. **理论贡献**：基于论文 1 的可观性边界，引入 Fisher 块矩阵解释几何与刚度何时可分、何时混淆。
3. **工程贡献**：输出四类维护决策：正常、几何可修、刚度退化需结构处置、复合病害需复查。
4. **可靠性贡献**：加入 UQ/校准/拒识机制，使模型不只给结果，也给“是否可信”的概率性判断。

### 方法论

建议采用“五模块”方法，而不是把所有东西揉成一个黑箱网络：

1. **条件输入模块**
   - 输入：车体、构架、轮对加速度，可选轮轨力；
   - 条件：速度、车辆参数、采样间距、运行工况；
   - 形式：$\mathbf{a}(t), v, \theta \mapsto$ latent observation representation。

2. **Branch-FNO 编码模块**
   - 从多通道时序响应提取频域/时域特征；
   - 延续 Sec2 中 PCNIO 的 Branch network。

3. **空间 Trunk + FiLM 条件模块**
   - 查询任意空间位置 $x$；
   - 用速度和车辆参数调制空间基函数；
   - 输出空间连续函数而不是固定网格分类。

4. **双输出解码模块**
   - 几何头：$\hat z_r(x)$；
   - 刚度头：$\hat \eta_k(x)$ 或 $\log \hat \eta_k(x)$；
   - 加入非负、平滑、边界 mask 约束。

5. **物理一致性与决策模块**
   - 正向一致性：$\hat z_r,\hat \eta_k$ 回代 VTCM 后应重构响应；
   - 决策分类器：基于几何超限和刚度退化阈值输出维护类别；
   - UQ：MC dropout / ensemble / conformal calibration 生成置信区间和拒识规则。

### 实验设计

1. **E1 数据集扩展**
   - 当前 144 windows 不够正式投稿；
   - 扩展到至少 1000-3000 windows；
   - 覆盖多 seed、多速度、多长度、软化/硬化/局部/长区段/复合病害。

2. **E2 单任务 vs 联合任务**
   - 几何-only；
   - 刚度-only；
   - 联合双输出；
   - 比较联合训练是否改善或损害两个任务。

3. **E3 物理约束消融**
   - data-only；
   - data + smoothness；
   - data + VTCM forward consistency；
   - data + Fisher/可观性权重。

4. **E4 多通道/多速度消融**
   - carbody only；
   - carbody + bogie；
   - carbody + bogie + wheelset；
   - single speed vs speed-conditioned vs multi-speed。

5. **E5 决策级验证**
   - NORMAL；
   - GEOMETRIC_TAMPING；
   - STIFFNESS_REPLACE；
   - COMPOUND_INSPECT；
   - 输出混淆矩阵、拒识率-准确率曲线。

6. **E6 UQ 与超限概率**
   - 几何超限概率；
   - 刚度退化概率；
   - 校准曲线、coverage、interval width。

### 章节结构

1. Introduction：从“哪里不平”提升到“为什么不平，以及能不能修”。
2. Problem formulation：联合逆向问题、几何-刚度可辨识性、维护决策定义。
3. Method：PCNIO-Joint、双输出结构、物理一致性、UQ/决策。
4. Dataset and experiments：从论文 1 基准继承数据，并扩展为训练/测试 benchmark。
5. Results：精度、消融、泛化、UQ、维护决策。
6. Discussion：可观性边界、部署条件、灾后保通和大修决策价值。

### 推荐目标

优先：CACAIE / Automation in Construction / MSSP。  
理由：这篇才是 AI + 基础设施工程 + 维护决策的完整逆向方法论文。

## 4. 原 A/B/C 内容如何归并

| 原规划 | 新归属 | 处理方式 |
|---|---|---|
| A：PCEQ/物理约束有效性 | 论文 1 为主，论文 2 方法消融复用 | PCEQ 从 benchmark 改为可观性-约束有效性诊断 |
| A：Fisher 信息/条件数 | 两篇都用 | 论文 1 解释正向可观性，论文 2 解释逆向可辨识性 |
| B：多速度融合/宽频可观性 | 论文 1 理论与边界，论文 2 扩展模块 | 不单独成篇，作为增强辨识的关键模块 |
| B：灾后保通叙事 | 论文 2 主工程场景 | 用于维护/决策价值，不宜放在论文 1 过重 |
| C：几何-刚度联合辨识 | 论文 2 主体 | 不再作为第三篇，作为第二篇核心贡献 |
| UQ/超限概率 | 论文 2 主体 | 放在模型可靠性与决策层 |
| 快速退化 PSD 演化 | 暂缓或作为论文 2 Discussion/后续 | 目前证据链不如几何-刚度联合辨识成熟 |

## 5. 新的核心缺口表述

建议以后统一使用下面这组 gap，而不是分散说“没有 PINO”“没有 UQ”“没有多速度”。

### Gap 1：可观性缺口

Existing onboard inverse methods often assume that track states are recoverable from vehicle responses, but rarely quantify the observability boundary imposed by vehicle suspension, speed, sensor placement and structural parameter perturbations.

中文：现有车载逆向方法通常默认轨道状态可由车辆响应恢复，却很少定量说明车辆悬挂、运行速度、传感器布置和结构参数扰动造成的可观性边界。

### Gap 2：约束有效性缺口

Physics-informed models are commonly reported as more accurate, but the mechanism by which physical constraints improve the conditioning of vehicle-track inverse problems remains unclear.

中文：物理信息模型常被报告为更准确，但物理约束究竟如何改善车轨逆问题条件数、在哪些工况下有效，仍缺少机制解释。

### Gap 3：根因辨识缺口

Most inverse mapping studies recover track geometry only, whereas maintenance decisions require distinguishing geometric defects from support stiffness degradation.

中文：多数逆向映射研究只恢复轨道几何，而维护决策真正需要区分几何病害与支撑刚度退化。

### Gap 4：可信决策缺口

Predicted track profiles are rarely accompanied by calibrated uncertainty or rejection rules, limiting their use in post-disaster and maintenance decisions.

中文：现有反演结果很少配套校准不确定性和拒识机制，限制了其在灾后保通和大修决策中的应用。

## 6. 近期最小可执行路线

### 先做论文 1 的最小闭环

1. 整理已有正向仿真结果：
   - 空吊；
   - 过渡段沉降；
   - 刚度-only；
   - 刚度-阻尼同步；
   - joint inverse 三案例。

2. 补一个可观性分析脚本：
   - 输入一个 sweep 的 npz；
   - 对不同病害/速度/通道计算响应增量 PSD 和 RMS；
   - 输出通道敏感性矩阵。

3. 补 Fisher/雅可比的轻量数值版本：
   - 对病害参数做有限差分；
   - 计算 $\partial a / \partial p$；
   - 画奇异值/条件数随波长、速度、通道变化。

4. 形成 4 张主图：
   - Figure 1：VTCM 基准与病害注入框架；
   - Figure 2：几何/空吊/沉降/刚度的响应敏感性图谱；
   - Figure 3：几何-刚度反事实混淆案例；
   - Figure 4：速度-波长-通道可观性边界。

### 再推进论文 2 的最小闭环

1. 扩展联合逆向数据集到正式规模。
2. 先训练两个轻量 baseline：
   - CNN/TCN/Transformer encoder + MLP decoder；
   - FNO/DeepONet style PCNIO。
3. 做双输出几何+刚度任务。
4. 加物理一致性和 UQ，而不是一开始就把所有模块堆满。
5. 先证明“联合辨识 + 决策分类”成立，再考虑多速度融合。

## 7. 需要推迟的内容

以下内容有价值，但现在不宜放进主线：

- PSD 时间演化与快速退化率预测：可以作为后续第三篇或论文 2 的扩展讨论；
- 多车次贝叶斯融合：适合灾后保通专题，但当前数据还未形成；
- CR450 385 km/h 大速度范围：可在论文 1/2 中作为外推或补充，不宜作为当前核心；
- 实际数据验证：有则加分，没有也不要让它成为主线依赖。

## 8. 最终推荐

最稳的两篇组合是：

1. **论文 1：Forward benchmark + observability boundary**
   - 先投 MSSP/VSD；
   - 目标是建立可信的仿真基准、可观性图谱和方法论空白；
   - 直接利用当前正向仿真代码和已有结果，完成度最高。

2. **论文 2：PCNIO-Joint inverse mapping + decision support**
   - 再投 CACAIE/AIC/MSSP；
   - 目标是提出真正的逆向方法；
   - 以几何-刚度联合辨识、UQ、维护决策为核心，不再拆第三篇。

这样安排后，论文 1 给论文 2 提供可观性边界和数据基准，论文 2 给论文 1 的方法论落地为可用逆向模型。两篇之间形成“为什么能反演”和“怎样可信反演”的闭环。
