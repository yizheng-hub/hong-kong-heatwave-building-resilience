# 香港热浪建筑能源服务韧性项目

这是香港热浪建筑韧性研究唯一的现行项目根目录。完整本地文件夹包含原始/提供方输入、冻结配置、现行代码、必要中间文件、已验收求解记录、全部正式结果、Panel数据和23张最终PNG图片。

同名private GitHub仓库是这一目录的可再分发镜像：包含代码、配置、紧凑准备数据、论文对齐结果表、Panel数据、完整绘图Notebook、最终图片和紧凑验收结果包。大型求解记录、EnergyPlus安装以及受提供方约束的原始数据只保留在本地完整根目录，并由`data/SOURCE_MANIFEST.csv`和`docs/PROJECT_CONTENTS.csv`记录。

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

在完整本地根目录中执行更严格的独立检查：

```bash
python src/run_pipeline.py --check-only --full-local --report docs/audit/UNIFIED_PROJECT_VALIDATION.json
```

该检查会核对原始输入、编译输入、求解输出、正式结果哈希、代码锁和23张图，全程EnergyPlus调用为0。默认绘图流程会在`output/figures/`生成23张PNG，更新`docs/DATA_DICTIONARY.csv`和面向可再分发文件的`SHA256SUMS.txt`，并再次执行校验。大体量本地输入和求解树通过完整本地库存与正式结果哈希检查，不在每次绘图时重复计算全部文件哈希。`plot/plot.ipynb`中每个图对应一个完整、可直接编辑和运行的单元格，数据读取、处理和全部样式均在该单元格内完成。

## 目录

- `config/`：冻结的事件、情景与评价定义
- `data/`：仅放原始/外部输入或提供方访问说明，不放任何输出
- `intermediate/`：整理后的映射、编译输入和已验收求解记录
- `result/`：正式结果、分析表和各Panel源数据
- `output/`：最终PNG图片
- `plot/`：完整绘图Notebook
- `software/`：本地固定版本的EnergyPlus运行环境
- `src/`：现行的数据准备、模拟、分析、校验和流程代码
- `docs/`：数据字典、来源、图件合同、项目清单与运行说明

手稿与Supplementary Information不放入本代码仓库。

代码或文件名中的`E8`表示八场热浪实验，并非软件版本号。`data/SOURCE_MANIFEST.csv`逐项说明HK-BFETD、HKO、HKUST等输入的真实本地位置与访问边界。代码采用MIT License；第三方数据和派生记录仍受原数据提供方条款约束，详见`docs/SOURCE_AND_LICENSES.csv`。
