# 13 QFAVAR 作者代码解读（Korobilis & Schröder 复现包）

> 对应 `main` 分支上传的作者复现代码：`Forecasting/`、`Structural/`、`README.txt`。
> 本文件讲清楚：①代码整体架构；②数据格式；③估计引擎(MCMC 六步)；④结构模块产出；⑤**与本文的差距(符号约束没放出来)**；⑥**为本文适配的清单**。

---

## 一、整体架构

| 模块 | 主程序 | 作用 |
|------|--------|------|
| **Forecasting/** | `FORECASTING_0_QFAVAR.m`（及 _1/_2/_3） | 递归伪样本外**预测评估**：QFAVAR vs FAVAR / 随机波动模型 / 单变量分位回归（原文第 4 节、表 2、图 3-4） |
| **Structural/** | `QFAVAR_GIRF.m`、`FAVAR_GIRFs.m` | **结构分析**：分位数因子估计 + **广义脉冲响应(GIRF)** + 方差分解(FEVD) + 国别投影（原文第 5 节、图 5、图 7） |
| 共用 | `functions/`、`data/` | 函数库 + 数据(`all_data.xlsx` + `load_data.m`) |

> **对本文最相关的是 `Structural/QFAVAR_GIRF.m`** —— 我们的实证主体（脉冲响应、国别异质性）以它为基底。Forecasting 模块可用于"QFAVAR 优于 FAVAR"的拟合优势论证（稳健性/卖点）。

> **依赖提醒**：`QFAVAR_GIRF.m` 用到 MATLAB **Econometrics Toolbox** 的 `armairf`（算 GIRF）。无该工具箱需替换（GIRF 可手写，或我们改用符号约束识别时本就不用它）。

---

## 二、数据格式与 `load_data.m`（适配的第一关）

`data/all_data.xlsx` 有**两个工作表**：

1. **`EA` 表（国别变量 x）**：
   - 第 1 行 = `tcode`（每列的平稳化变换代码，供 `transxFAVAR` 用）；第 2 行起 = 数据；第 6 行起的首列 = 日期。
   - **列按"变量大块 → 块内 9 国"排列**：每个宏观变量占连续 9 列（9 个欧元区国家）。
   - `inflindx` 选通胀口径（HICP 总/去能源/核心）；`outpindx` 在失业率↔工业生产间切换。
   - 删除 NPCR 后保留 **45 列 = 5 变量 × 9 国**。
   - `transxFAVAR(x,dates,tcode)` 按 tcode 做平稳化。
2. **`Global` 表（全球观测变量 g）**：4 个全球因子（GINF/GECON、GSCPI、FCI、GEPU），已是平稳形式。

关键设置：`standar=2` → 对 **x 和 g 都做 z-score 标准化**；`dfm=1` → 令 `g=[]`（退化为只用 x 的分位数因子模型 QDFM）。

> **⚠️ 硬编码"9"（国家数）遍布全代码**——这是我们 16 国适配最容易出错的地方（见第六节）。

---

## 三、核心维度与符号（读代码必备对照表）

| 符号 | 含义 | 欧元区取值 | 本文取值 |
|------|------|-----------|----------|
| `n` | 国别序列数 | 45（5×9） | **64（4×16）** |
| `r` | 因子数 = 宏观变量块数 | 5 | **4**(CPI/IP/利率/NEER) |
| `nq` | 分位数个数 | 3（.1/.5/.9） | 3 |
| `ng` | 全球观测变量数 | 4 | **5**(产量/IGREA/油价/GSCPI/GEPU) |
| 状态 VAR 维度 | `r*nq+ng` | **19** | **4×3+5=17** |
| `p` | 滞后阶 | 2 | 2 |
| `k` | 每方程 VAR 参数 | `(r*nq+ng)*p+interF` | 同式 |

---

## 四、估计引擎：MCMC 六步（Structural 与 Forecasting 共用）

**起点（两步法）**：先用 `extract`（PCA 取第一主成分）和 `VBQFA`（变分贝叶斯分位数因子，Korobilis-Schröder 2022）对每个变量块估出分位数因子作为初值/插值（`QFAVAR_GIRF.m` 第 107-111 行）。

**Gibbs 主循环**（每次迭代）：

| 步 | 代码位置 | 做什么 |
|----|----------|--------|
| 1 | 测量方程逐方程 | **抽载荷 L**（horseshoe 先验 + `randn_gibbs` 高维高斯后验） |
| 2 | 同上 | **抽潜变量 z**（把非对称拉普拉斯写成正态尺度混合，逆高斯抽样；`ALsampler` 选 Khare-Hobert 或 Kozumi-Kobayashi） |
| 3 | 同上 | **抽尺度 Sigma**（逆 Gamma） |
| — | `(ir-1)*9+1:ir*9` | **载荷归一化**：把每个因子第 q 个国家的载荷设为 1（固定符号与尺度） |
| 4 | `FFBS(...)` | **抽因子 [F;g]**（Carter-Kohn 前向滤波后向抽样 = 模拟平滑器） |
| — | `sign(Ctemp)` | **因子符号对齐**：以 VBQFA 为参照，保证同一因子在不同分位数符号一致 |
| 5 | `var_sv` 分支 | **抽 VAR 方差 Omega**（`var_sv=0` 常数逆 Gamma；`=1` 随机波动 SVRW）→ QFAVAR vs QFAVAR-SV |
| 6 | `randn_gibbs` + `while max(abs(eig(Phic)))>0.999` | **抽 VAR 系数 Phi**（horseshoe；**只接受平稳解**，拒绝爆炸 VAR） |

> 这六步精确对应原文 2.3 节，也对应我们 `03_QFAVAR方法详解.md` 讲的内容。`var_sv` 开关 = QFAVAR-SV。

---

## 五、结构模块产出（`QFAVAR_GIRF.m` 后半段）

烧入后、每 `nthin` 次保存一次，并计算：

1. **因子 GIRF**：`firf = armairf(ar0,[],'InnovCov',OMEGA,'Method','generalized',...)` —— **广义脉冲响应**（不依赖排序、非结构识别）。
2. **国别 GIRF（投影）**：`yirf = [firf...]*LL'` —— 通过载荷矩阵 `LL` 把因子层面响应**投影回 64 个国别序列 × 3 分位数**。这正是我们要的"国别异质性"机制。
3. **方差分解 FEVD**：`fevd = cumsum(firf.^2)./sum(...)` —— 各冲击对各分位数因子波动的贡献占比（原文图 7）。
4. **绘图**：分位数因子时序图、各全球冲击的因子 IRF（10/50/90 三线）、FEVD 面积图。
5. **保存**：`save('QFAVAR_GIRFs.mat')`。

---

## 六、★ 与本文的关键差距：符号约束没有放出来

- 发布代码的结构识别 = **GIRF（广义脉冲响应）**，对应原文**图 5（Section 5.1）**。
- 原文**图 6 / 表 3（Section 5.2）的"符号 + 弹性约束"结构识别，源码里没有**。
- 函数库里虽有 `drawBVARsign.m`（带 horseshoe 的 BVAR + 因子/冲击抽样，具备做符号约束的骨架），但 `QFAVAR_GIRF.m` **并未调用它**。

**这对本文意味着**（重要）：
1. **GIRF 可直接复用** → 用于我们 `5.1 节"展示异质性"`（和原文图 5 同款），快速看到分位数发散。
2. **我们的基准识别（符号约束 + 供给弹性约束）需要自己加**：在 `QFAVAR_GIRF.m` 已有的后验抽样 `OMEGA_draws`/`Phi_draws` 之上，加一段 **RWZ(2010) 旋转 + 油市块符号/弹性检验**（见 `07`），接受的旋转再算结构 IRF、国别投影、FEVD。`drawBVARsign.m` 的旋转/抽样骨架可作参考。
3. 工作量可控：估计引擎（最难部分）现成，我们主要**替换"GIRF 计算"为"符号约束结构 IRF 计算"**。

---

## 七、Forecasting 模块（备用于"QFAVAR 优于 FAVAR"卖点）

- `FORECASTING_0_QFAVAR.m`：递归伪样本外预测（`first_per=0.5` 起始样本，逐期扩展，预测 h=24 步），用**分位数得分(Quantile Score)**、预测似然、PIT 评估，比较 QFAVAR 与各基准。
- 同一 MCMC 引擎，外面套一个"逐期重估 + 预测"的循环。
- 对本文：可选做一节"QFAVAR 相对 FAVAR 的尾部预测优势"，呼应原文表 2，增强方法说服力（属稳健性/加分，非主线）。

---

## 八、函数库速查（`functions/`）

| 函数 | 作用 |
|------|------|
| `load_data.m` | 读 xlsx、选口径、平稳化、标准化 |
| `transxFAVAR.m`/`transxf.m` | 按 tcode 做平稳化变换 |
| `extract.m` | PCA 取主成分（因子初值） |
| `VBQFA.m` | **变分贝叶斯分位数因子**（两步法核心，秒级） |
| `FFBS.m`/`FFBS_rob.m`/`FFBSVB.m` | 前向滤波后向抽样（抽因子） |
| `randn_gibbs.m` | 高维高斯回归系数高效抽样 |
| `horseshoe.m`/`horseshoe_prior.m` | 马蹄先验更新 |
| `drawBVARsign.m` | 带因子结构的 BVAR 抽样（**可改造做符号约束**） |
| `SVRW.m` | 随机波动(Chan 滤波) |
| `gigrnd.m`/`trandn.m`/`mvnrnd.m` | 各类随机数(广义逆高斯/截断正态/多元正态) |
| `mlag2.m`/`SURform.m`/`vec.m`/`mfc_con_est_new.m`/`update_Omega.m` | VAR/矩阵工具 |
| `BN_IC.m`/`NbFactors.m`/`nbpiid.m` | 因子个数信息准则(Bai-Ng) |
| `armairf`(MATLAB 内置) | 广义脉冲响应 |
| `shade.m`/`shadedplot.m`/`suptitle.m` | 绘图 |

---

## 九、★ 为本文适配的清单（动手前对照）

| # | 适配项 | 具体改动 |
|---|--------|----------|
| 1 | **数据替换** | 把 `all_data.xlsx` 的 `EA` 表换成我们 16 国 × 4 变量（CPI/IP/利率/NEER，列序=变量块→块内16国）；`Global` 表换成 5 个全球变量(产量/IGREA/油价/GSCPI/GEPU) |
| 2 | **硬编码 9 → 16** | 全局把"国家数 9"改 16：`(ifac-1)*9+1:ifac*9`、`fqfa`、载荷归一化 `(ir-1)*9`、符号对齐、`floor((i-1)/9+1)`、`reshape(...,9,r)`、`names([1:n/r:n])`、绘图网格等。**逐处核对，最易出错** |
| 3 | **r、ng** | `r=4`（我们 4 个宏观变量）；`ng=5`（5 个全球变量） |
| 4 | **平稳化 tcode** | 我们数据已是同比/增长率等平稳形式 → 把 tcode 设为"不变换"(或跳过 `transxFAVAR`)，避免二次变换 |
| 5 | **口径开关** | `inflindx`/`outpindx` 是欧元区专用 → 简化或删除，直接用我们的 4 变量 |
| 6 | **★ 加符号约束识别** | 在后验抽样上加 RWZ 旋转 + 油市块(产量/IGREA/油价)符号约束 + 供给弹性上界(见 `07`)；GSCPI/GEPU 作控制不约束；分位数因子全自由。替换原 `armairf generalized` 这一步 |
| 7 | **标准化** | 保持 `standar=2`（x、g 都标准化）；注意我们数据有缺失，需确认 `zscore`/估计对 NaN 的处理（或先按 `11` 的方案补缺失） |
| 8 | **缺失值** | 一步式 MCMC 的 `FFBS` 可处理缺失；若 VBQFA 起点要求平衡面板，则先插值(见 `11`) |

---

## 十、建议的跑通步骤

1. **原样复现**：在作者 `all_data.xlsx` 上跑通 `QFAVAR_GIRF.m`，确认环境(含 Econometrics Toolbox)与出图正常。
2. **换数据**：按【九】1-5 替换数据、改 9→16、设 r/ng/tcode，先得到 **GIRF 版**结果（分位数因子、因子 GIRF、国别投影、FEVD）。
3. **加识别**：按【九】6 实现符号约束 + 弹性约束，得到**基准结构结果**。
4. **稳健性**：递归(Cholesky)、叙事、Känzig；Forecasting 模块做拟合优势。

---

*这是代码层面的"地图"。下一步我可以：①帮你写"数据 → all_data.xlsx 两表格式"的转换脚本；②帮你把 9→16 等适配点逐条改好；③把符号约束识别那段代码写出来接到后验抽样上。*
