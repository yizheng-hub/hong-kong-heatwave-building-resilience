# 香港热浪建筑能源服务韧性项目

本仓库包含来源输入清单、建筑—事件结果、分析工具和可编辑绘图Notebook，用于复现项目最终科学结果与23张静态图片。研究内容涵盖香港建筑在热浪中的电力需求、制冷服务缺口、室内过热和恢复表现。

仓库能够独立核对所发布的结果表，并重新生成6张正文图和17张补充图。它不会重新调用EnergyPlus：已经验收的建筑—事件结果和城市时间序列作为分析输入随仓库提供；原始求解器输出体量很大，部分观测比较数据也受提供方许可限制，因此不在GitHub中重新分发。

## 快速运行

```bash
git lfs install
conda env create -f environment.yml
conda activate hk-heatwave
python -m ipykernel install --user --name hk-heatwave --display-name "HK heatwave"
python src/run_pipeline.py
```

只检查输入而不重新绘图：

```bash
python src/run_pipeline.py --check-only
```

完整流程会在`output/figures/`生成23张PNG，更新`docs/DATA_DICTIONARY.csv`和`SHA256SUMS.txt`。`plot/plot.ipynb`中每个图对应一个完整、可直接编辑和运行的单元格，数据读取、处理和全部样式均在该单元格内完成。

## 目录

- `config/`：冻结的事件、情景与评价定义
- `data/`：仅放原始/外部输入或提供方访问说明，不放任何输出
- `intermediate/`：整理后的建筑、空间映射与城市时间序列
- `result/`：建筑—事件结果、分析表和各Panel源数据
- `output/`：最终PNG图片
- `plot/`：完整绘图Notebook
- `src/`：输入检查、分析指标和流程入口
- `docs/`：数据字典、来源、图件合同与运行说明

手稿与Supplementary Information不放入本代码仓库。

`data/SOURCE_MANIFEST.csv`逐项说明HK-BFETD、HKO、HKUST等输入的本地位置与访问边界。代码采用MIT License；第三方数据和派生记录仍受原数据提供方条款约束，详见`docs/SOURCE_AND_LICENSES.csv`。
