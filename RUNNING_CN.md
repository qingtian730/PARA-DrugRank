# PARA-DrugRank 运行说明

本仓库提供三个细胞系任务及 BeatAML 纵向患者任务的训练和评价代码。真实数据由使用者在本地准备，训练会生成新的模型权重与留出预测。数据字段和数组形状见 [README](README.md)。

## 先运行合成示例

在 Python 3.12 虚拟环境中，从仓库根目录运行下列命令。示例会构造合成数据，拟合四个细胞系专家，并检查模型保存、加载和预测。

```sh
python -m pip install -r requirements.txt
python source/demo.py
python -m unittest discover -s tests -v
```

## 训练细胞系模型

将本地准备的输入分别命名为 `prism.npz`、`gdsc2.npz` 或 `ctrpv21.npz`。每个文件包含按轴对齐的表达、来源药敏与目标药敏，以及固定的外层五折和内层四折划分。下例训练并评价 PRISM 第0折，最后一条命令评估全部五折。

```sh
python source/train.py --data data/processed/prism.npz --outer-fold 0 --output outputs/prism_fold0.joblib
python source/evaluate.py --data data/processed/prism.npz --checkpoint outputs/prism_fold0.joblib
python source/run_all_folds.py --cohort prism --data-dir data/processed --output-dir outputs/prism_cv
```

默认在每个外层训练集内重新选择专家、分数变换和组合权重。默认比较面板分别包含105、100和92种药物。用 GDSC2 的101种药物完整输入运行默认比较时，程序会移除 `iap5620`。要复现完整面板实验，请在独立环境中安装 `requirements-full-panel.txt`，训练与评价时均指定 `--panel full`。

## 训练患者模型

另建 Python 3.12 环境。每一行对应一名患者，早期表达、早期和后期离体药敏按患者及药物对齐，同一患者的配对样本在每次划分中进入同一折。

```sh
python -m pip install -r requirements-patient.txt
python source/train_patient.py --data data/processed/beataml.npz --output-dir outputs/beataml
```

训练程序在内层比较73个候选模型，以患者平均 Spearman 相关选择模型，并在留出患者上评价后期排序。请为每次运行指定新的输出路径。
