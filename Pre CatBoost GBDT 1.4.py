#by miaomiao lu
# ===============================================================
# 光谱数据建模（支持多处理Excel表：MSC / SNV / 一阶导数）
# 自动导出性能汇总表 + 合并三处理对比图 + 特征波长提取
# ===============================================================

import os
import tempfile
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
# 导入GBDT模型
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = ["SimHei", ]
plt.rcParams['axes.unicode_minus'] = False


# ------------------- 基础环境配置 -------------------
def set_safe_temp_dir():
    base_temp = os.environ.get('TEMP', '/tmp')
    try:
        base_temp.encode('ascii')
        safe_temp = os.path.join(base_temp, 'spectral_ml_temp')
    except UnicodeEncodeError:
        safe_temp = 'C:\\spectral_ml_temp' if os.name == 'nt' else '/tmp/spectral_ml_temp'

    os.makedirs(safe_temp, exist_ok=True)
    os.environ['TMPDIR'] = safe_temp
    os.environ['TEMP'] = safe_temp
    tempfile.tempdir = safe_temp
    return safe_temp


def configure_joblib_backend():
    joblib.parallel_backend('sequential')


safe_temp_dir = set_safe_temp_dir()
configure_joblib_backend()
print(f"已配置安全临时目录: {safe_temp_dir}")


# ------------------- 光谱分类核心类 -------------------
class SpectralClassifier:
    def __init__(self, excel_path, sheet_name, weight_path, output_dir):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.weight_path = weight_path
        self.output_dir = output_dir
        self.X = None
        self.y = None
        self.sample_weights = None
        self.wavelengths = None
        # 更新类别标签为纯数字，与实际映射一致
        self.classes = np.array(['0', '1', '3', '5', '7', '9'])
        self.models = {}
        os.makedirs(output_dir, exist_ok=True)
        self.plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        self.performance_records = []
        # 存储特征重要性相关信息
        self.feature_names = None
        self.importance_records = []

    def load_weight_mapping(self):
        # 读取更新后的权重文件
        weight_df = pd.read_excel(self.weight_path)
        if '序号' not in weight_df.columns or '权重' not in weight_df.columns:
            raise ValueError("权重文件中必须包含 '序号' 和 '权重' 列")
        mapping = dict(zip(weight_df['序号'].astype(int), weight_df['权重'].astype(float)))
        print(f"成功读取 {len(mapping)} 个权重记录。")
        return mapping

    def load_data(self):
        print(f"\n正在读取工作表：{self.sheet_name}")
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        weight_mapping = self.load_weight_mapping()

        # 处理编号列（支持“3 24H”或“3”或含多空格情况）
        id_col = df.iloc[:, 0].astype(str).str.strip()

        # 将空格数量不定的字符串拆分成两列（多余忽略）
        split_cols = id_col.str.split(r'\s+', n=1, expand=True)

        # 如果只有一列（没有时间），补一个空字符串
        if split_cols.shape[1] == 1:
            split_cols[1] = "0H"

        split_cols.columns = ['序号', '时间']
        # 提取数字部分的“序号”列
        df['序号'] = split_cols['序号'].astype(str).str.extract(r'(\d+)')
        df['序号'] = pd.to_numeric(df['序号'], errors='coerce').fillna(0).astype(int)

        # 清理“时间”列（移除H、空格、None、非数字）
        df['时间'] = split_cols['时间'].astype(str).str.replace('H', '', regex=False).str.strip()
        df['时间'] = pd.to_numeric(df['时间'], errors='coerce').fillna(0).astype(float)

        # 删除原始第一列
        df = df.drop(df.columns[0], axis=1)

        # 筛选出能转换为浮点数的列名作为波长列
        wavelength_columns = []
        for col in df.columns:
            try:
                float(col)  # 尝试转换为浮点数
                wavelength_columns.append(col)
            except ValueError:
                continue  # 跳过不能转换的列名

        # 提取波长列并转换为浮点数
        self.wavelengths = np.array(wavelength_columns).astype(float)

        # 收集有效样本的索引（同步过滤X、y、sample_weights）
        valid_indices = []
        y = []
        sample_weights = []
        for idx, seed_num in enumerate(df['序号']):
            # 更新序号到类别的映射规则
            if 1 <= seed_num <= 55:  # 1-55 对应 0
                cls = '0'
            elif 56 <= seed_num <= 115:  # 56-115 对应 1
                cls = '1'
            elif 116 <= seed_num <= 175:  # 116-175 对应 3
                cls = '3'
            elif 176 <= seed_num <= 235:  # 176-235 对应 5
                cls = '5'
            elif 236 <= seed_num <= 285:  # 236-285 对应 7
                cls = '7'
            elif 286 <= seed_num <= 345:  # 286-345 对应 9
                cls = '9'
            else:
                continue  # 跳过无效样本
            # 记录有效样本的索引、标签和权重
            valid_indices.append(idx)
            y.append(cls)
            sample_weights.append(weight_mapping.get(seed_num, 1.0))

        # 只保留有效样本的特征数据（通过索引过滤）
        X = df.iloc[valid_indices][wavelength_columns].to_numpy()

        self.X = np.array(X)
        self.y = np.array(y)
        self.sample_weights = np.array(sample_weights)
        print(f"数据加载完成：{self.X.shape[0]}个样本，{self.X.shape[1]}个波长特征。")
        print(f"类别分布：{pd.Series(self.y).value_counts().sort_index()}")
        return self

    def check_class_balance(self, plot=True):
        class_counts = pd.Series(self.y).value_counts().reindex(self.classes)
        print("\n类别分布：")
        print(class_counts)
        if plot:
            plt.figure(figsize=(8, 5))
            sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
            plt.title(f'{self.sheet_name} 类别分布')
            plt.xlabel('类别')
            plt.ylabel('样本数量')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{self.sheet_name}_class_dist.png'), dpi=300)
            plt.close()

    def handle_class_imbalance(self):
        class_weights = compute_class_weight('balanced', classes=self.classes, y=self.y)
        class_weight_dict = dict(zip(self.classes, class_weights))
        print("类别权重:", class_weight_dict)
        return self.X, self.y, self.sample_weights, class_weight_dict

    def feature_engineering(self, X):
        # 保存原始特征名称（用于后续特征重要性分析）
        n_wavelengths = len(self.wavelengths)

        # 计算导数特征
        first_deriv = np.gradient(X, axis=1)
        second_deriv = np.gradient(first_deriv, axis=1)

        # 统计特征
        mean_vals = np.mean(X, axis=1).reshape(-1, 1)
        std_vals = np.std(X, axis=1).reshape(-1, 1)
        # 修复未定义stats_features的错误
        stats_features = np.hstack([mean_vals, std_vals])

        # 构建特征名称列表
        self.feature_names = []
        self.feature_names.extend([f"原始_{w}" for w in self.wavelengths])
        self.feature_names.extend([f"一阶导数_{w}" for w in self.wavelengths])
        self.feature_names.extend([f"二阶导数_{w}" for w in self.wavelengths])
        self.feature_names.extend(["均值", "标准差"])

        # 拼接所有特征
        features = np.hstack([X, first_deriv, second_deriv, stats_features])
        return StandardScaler().fit_transform(features)

    def get_top_wavelength_importance(self, model, model_name, top_n=15):
        """提取并分析前N个重要的波长特征"""
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature_name': self.feature_names,
                'importance': importances
            })

            # 筛选出原始波长特征
            original_wavelength_df = importance_df[
                importance_df['feature_name'].str.startswith('原始_')
            ].copy()

            # 提取波长数值
            original_wavelength_df['wavelength'] = original_wavelength_df['feature_name'].str.replace('原始_',
                                                                                                      '').astype(float)

            # 按重要性排序
            original_wavelength_df = original_wavelength_df.sort_values('importance', ascending=False)

            # 获取前N个重要波长
            top_wavelengths = original_wavelength_df.head(top_n)

            print(f"\n{self.sheet_name} - {model_name} 前{top_n}个重要特征波长：")
            print("-" * 50)
            for idx, row in top_wavelengths.iterrows():
                print(f"波长 {row['wavelength']:.1f} nm - 重要性: {row['importance']:.4f}")

            # 保存到记录中
            self.importance_records.append({
                "预处理": self.sheet_name,
                "模型": model_name,
                "前15波长": ', '.join([f"{w:.1f}" for w in top_wavelengths['wavelength'].values]),
                "对应重要性": ', '.join([f"{i:.4f}" for i in top_wavelengths['importance'].values])
            })

            # 绘制特征重要性图
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_wavelengths, x='wavelength', y='importance', palette='coolwarm')
            plt.title(f'{self.sheet_name} - {model_name} 前{top_n}个重要波长特征', fontsize=12)
            plt.xlabel('波长 (nm)', fontsize=10)
            plt.ylabel('特征重要性', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plot_dir, f"{self.sheet_name}_{model_name}_top{top_n}_wavelength_importance.png"),
                dpi=300)
            plt.close()

            return top_wavelengths

        return None

    def optimize_and_evaluate(self, X_train, X_test, y_train, y_test, train_weights, class_weights):
        # GBDT模型
        models = {
            "CatBoost": CatBoostClassifier(
                objective='MultiClass',
                eval_metric='Accuracy',
                verbose=0,
                random_state=42
            ),
            "GBDT": GradientBoostingClassifier(
                loss='log_loss',  # 多分类损失函数
                n_estimators=100,  # 树的数量
                learning_rate=0.1,  # 学习率
                max_depth=3,  # 树的最大深度
                random_state=42,
                verbose=0
            )
        }

        for name, model in models.items():
            # GBDT需要特殊处理样本权重
            if name == "GBDT":
                model.fit(X_train, y_train, sample_weight=train_weights)
            else:
                model.fit(X_train, y_train, sample_weight=train_weights)

            # 提取并显示前15个重要波长
            self.get_top_wavelength_importance(model, name, top_n=15)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.classes, output_dict=True)
            cm = confusion_matrix(y_test, y_pred, labels=self.classes)

            cm_path = os.path.join(self.plot_dir, f"{self.sheet_name}_{name}_cm.png")
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
            plt.title(f"{self.sheet_name} {name} 混淆矩阵")
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()

            print(f"{self.sheet_name} - {name} 准确率: {acc:.4f}")

            self.performance_records.append({
                "预处理": self.sheet_name,
                "模型": name,
                "准确率": acc,
                "混淆矩阵文件": cm_path, **{f"{cls}_F1": report[cls]["f1-score"] for cls in self.classes}
            })

        # 导出Excel性能汇总表
        summary_path = os.path.join(self.output_dir, "performance_summary.xlsx")
        pd.DataFrame(self.performance_records).to_excel(summary_path, index=False)

        # 导出特征重要性表
        if self.importance_records:
            importance_path = os.path.join(self.output_dir, "feature_importance_summary.xlsx")
            pd.DataFrame(self.importance_records).to_excel(importance_path, index=False)
            print(f"✅ {self.sheet_name} 特征重要性结果已保存到：{importance_path}")

        print(f"✅ {self.sheet_name} 性能结果已保存到：{summary_path}")


# ------------------- 主程序入口 -------------------
if __name__ == "__main__":
    # 光谱数据文件路径
    excel_path = r"D:\APP\Python\project\2025.10.19 ZD 15 013579 96H\0H光谱数据预处理结果.xlsx"
    # 更新权重文件路径为O2-浓度.xlsx
    weight_path = r"G:\10.11-10.15ZD 15 013579\O2-浓度.xlsx"

    sheets = {
        "MSC": "MSC 处理后",
        "SNV": "SNV 处理后",
        "DERIV": "一阶导数处理后"
    }

    all_results = []
    all_importance_results = []

    for tag, sheet in sheets.items():
        print(f"\n{'=' * 60}\n开始处理 {sheet} 数据\n{'=' * 60}")
        output_dir = f"TZspectral_model_results_with_weights_{tag}"
        classifier = SpectralClassifier(excel_path, sheet, weight_path, output_dir)
        classifier.load_data()
        classifier.check_class_balance()
        X, y, w, cw = classifier.handle_class_imbalance()
        X_feat = classifier.feature_engineering(X)
        X_train, X_test, y_train, y_test, train_w, test_w = train_test_split(
            X_feat, y, w, test_size=0.3, random_state=42, stratify=y
        )
        classifier.optimize_and_evaluate(X_train, X_test, y_train, y_test, train_w, cw)
        all_results.extend(classifier.performance_records)
        all_importance_results.extend(classifier.importance_records)

    # ------------------- 汇总所有结果 -------------------
    summary_dir = "1.4 0H TZspectral_model_results_with_weights_summary"
    os.makedirs(summary_dir, exist_ok=True)

    # 保存性能汇总
    df_summary = pd.DataFrame(all_results)
    summary_excel = os.path.join(summary_dir, "overall_performance_summary.xlsx")
    df_summary.to_excel(summary_excel, index=False)

    # 保存特征重要性汇总
    if all_importance_results:
        df_importance = pd.DataFrame(all_importance_results)
        importance_excel = os.path.join(summary_dir, "feature_importance_summary.xlsx")
        df_importance.to_excel(importance_excel, index=False)
        print(f"\n✅ 特征重要性汇总结果保存到：{importance_excel}")

    # 准确率对比图
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_summary, x="预处理", y="准确率", hue="模型", palette="viridis")
    plt.title("三种预处理模型准确率对比 (CatBoost vs GBDT)")
    plt.ylabel("Accuracy")
    plt.xlabel("预处理方法")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()

    print(f"\n✅ 全部三种处理完成，汇总结果保存到：{summary_excel}")
    print(f"📊 对比图文件：{os.path.join(summary_dir, 'accuracy_comparison.png')}")