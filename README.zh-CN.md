# 清洁水产饲料配方筛选

[English](README.md)

论文 *A cleaner aquafeed formulation screening framework under nutritional, economic and environmental constraints for largemouth bass* 的代码。

框架以鱼粉含量 45% 的最低成本配方为基准，筛选大口黑鲈（*Micropterus salmoides*）鱼种阶段（10 至 50 g）的饲料配方，步骤如下：

1. 用线性规划制定鱼粉含量 35%、45% 和 55% 的最低成本参考配方；
2. 以不同目标权重下的 LP 顶点及其随机凸组合生成可行候选配方；
3. 相对基准计算成本和六项 ReCiPe 2016 中点（H）指标，施加负担转移惩罚，按四条判定规则分类；
4. 检验分类结果对规则阈值、原料价格不确定性和 LCA 系数不确定性的稳健性；
5. 与直接 LP、等能量 LP 以及改编自 Garcia-Launay et al. (2018) 的加权和方法比较；
6. 用生物能量学模型预测参考配方和稳健候选配方从 10 g 长到 50 g 所需天数和 FCR；
7. 将成本和环境影响的每项变化以及每条未通过的规则精确分解到各原料。

结果分三个层级：

| 层级 | 单位 |
|---|---|
| 原料层 | 每 kg 原样原料在出厂时的生命周期系数 |
| 配方层 | 1 kg 原样饲料中原料的上游影响 |
| 预测生产层 | 配方层影响乘以预测 FCR，即每 kg 预测增重的影响 |

## 安装

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 运行示例

```bash
python scripts/run_pipeline.py --config config/example.toml --out results_example
python -m pytest -q
```

`config/example.toml` 使用 `data/example/` 中的示例数据和较小的样本量，几分钟内完成。示例原料数据库是合成数据：价格和生命周期系数经过随机缩放，成分经过小幅扰动，结果与论文不同。

运行论文案例时，将原料数据库保存为 `data/ingredient_database.csv`（列与示例文件相同），然后运行

```bash
python scripts/run_pipeline.py --config config/default.toml --out results
```

可用 `--stages formulation uncertainty benchmarks growth figures tables` 分步运行，后面的步骤读取前面步骤的输出。所有随机步骤的种子都在配置文件中固定。

## 目录结构

```
config/default.toml          案例的全部设置（营养限值、规则、随机种子、样本量）
config/example.toml          使用示例数据和较小样本量的同一设置
data/example/                合成示例数据
src/aquafeed_screen/
    data.py                  原料数据库的读取与校验
    formulation.py           LP 模型、参考配方、LP 顶点、凸组合（式 1–12）
    evaluation.py            比值、负担转移惩罚、清洁评分、规则 R1–R4（式 13–20）
    uncertainty.py           价格与 LCA 蒙特卡洛、阈值敏感性（式 21–22）
    benchmarks.py            直接 LP、等能量 LP、加权和前沿、添加量上限
    growth.py                生物能量学生长模型（式 23–25）
    adc_check.py             数据库消化率与大口黑鲈实测值的比较
    attribution.py           原料贡献及规则未通过原因的归因（式 26–27）
    figures.py, reporting.py 图和补充数据表
    pipeline.py              全部步骤的运行
scripts/run_pipeline.py      命令行入口
tests/                       一致性检验（使用示例数据）
docs/methods.zh-CN.md        公式及其实现
```

## 输入数据

原料数据库每行一种原料，各列含义见英文 README 的表格。缺少必需数值、消化率超出 [0, 1] 或基准价格超出价格区间时，程序报错停止。`data/example/adc_largemouth_bass.csv` 为大口黑鲈实测消化率，摘自 `adc_largemouth_bass_references.csv` 所列文献，仅用于核对数据库数值。

用于其他鱼种、生长阶段或市场时，修改原料数据库及配置文件中的 `[nutrients]`、`[reference]`、`[candidates]` 和 `[growth]` 部分。生长模型参数仅适用于大口黑鲈。

## 引用

请引用论文（见 `CITATION.cff`）。

## 许可

PolyForm Noncommercial License 1.0.0，见 `LICENSE`。允许用于科研、教学等非商业用途。如有问题，请提交 issue。
