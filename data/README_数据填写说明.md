# data/ 数据填写说明

本目录用于存放论文实证所需数据。当前包含：

## 1. `国家特征_第二阶段_模板.csv`
用于"第二阶段分析"（见 `论文初稿/核心章节_模型与识别.md` 式 (5)）的国家结构特征。
**分类变量已预填**（发达/新兴、油气净进出口地位、是否通胀目标制、汇率制度），**定量列留空待你填写**。

### 各列含义与数据来源

| 列名 | 含义 | 取值/单位 | 数据来源 |
|------|------|-----------|----------|
| `code` / `country_cn` / `country_en` | 国家代码与名称 | — | 已填 |
| `advanced_emerging` | 发达 / 新兴 | advanced / emerging | 已填（IMF 分类） |
| `oil_net_position` | 石油净进出口地位 | importer / exporter / balanced | 已填，建议再用定量列 `oil_import_dependence` 替代 |
| `inflation_targeting` | 是否实行通胀目标制 | 1 / 0（含起始年备注） | 已填，核对：各国央行、IMF AREAER |
| `fx_regime` | 汇率制度 | float / managed / peg | 已填，核对：IMF AREAER、Ilzetzki-Reinhart-Rogoff 分类 |
| `cpi_energy_transport_share` | CPI 中能源+交通权重 | % | **待填**：OECD、各国统计局、Eurostat（COICOP 分项权重） |
| `fossil_fuel_subsidy_gdp` | 化石燃料补贴/GDP | % 或指数 | **待填**：IMF Fossil Fuel Subsidies 数据库、IEA |
| `oil_import_dependence` | 石油净进口依赖度 | 净进口/消费 或 净进口额/GDP | **待填**：IEA、EIA、World Bank WDI、Energy Institute Statistical Review |
| `energy_intensity` | 能源强度 | 一次能源/GDP | **待填**：World Bank WDI、IEA |
| `cbi_index` | 央行独立性指数 | 0—1 | **待填**：Garriga (2016) CBI 数据集、Romelli (2022) |
| `trade_openness` | 贸易开放度 | (进口+出口)/GDP，% | **待填**：World Bank WDI |
| `financial_dev_credit_gdp` | 金融发展 | 私人信贷/GDP，% | **待填**：World Bank WDI、IMF Financial Development Index |
| `notes` | 备注 | — | 已填关键提示 |

### 填写建议
- 定量特征多为低频（年度）结构变量；可取**样本期均值**作为该国的一个截面取值。
- 个别国家（阿根廷、土耳其、俄罗斯）需在 `notes` 注明特殊处理。
- 数据来源务必在论文附录列明、可复现。

## 2. 还需建立的数据文件（建议命名）
- `g20_cpi_monthly.csv`：19 国月度 CPI（用于计算通胀率）— OECD / IMF IFS / FRED
- `g20_macro_monthly.csv`：工业生产、短期利率、名义有效汇率 — OECD / BIS
- `global_oil_block.csv`：实际油价（Brent）、世界石油产量、全球经济活动指数（Kilian / Baumeister-Hamilton）、OECD 石油库存 — EIA / FRED / 作者主页
- `kanzig_oil_supply_news.csv`：Känzig (2021) OPEC 工具序列 — GitHub `dkaenzig/oilsupplynews`（稳健性识别用）

> 需要我写 Python 抓取脚本（OECD/FRED/EIA API 批量下载 + 清洗 + 季调）时，告诉我即可。
