import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== 1. 读取你的二维数据 =====================
# 方法1：直接读取Excel（推荐）
file_path = r"G:\10.11-10.15ZD 15 013579\相关分析热图数据.xlsx"
# 你的数据结构：第一行是列名(A0/A1/A3/A5/A7/A9)，第一列是行名(发芽势/发芽率等)
df = pd.read_excel(
    file_path,
    header=0,    # 第一行作为列名
    index_col=0  # 第一列作为行索引
)

# 方法2：如果Excel读取有问题，可直接粘贴数据（备用）
# data = {
#     'A0': [80, 90.9, 22.70109091, 2.170545455, 26.35362424, 6160.633333, 6204.368167, 6768.247, 1597.343667, 6118.55],
#     'A1': [53.33, 81.67, 17.927, 1.923166667, 34.73368889, 6257.994167, 6293.024833, 6705.130333, 1672.486517, 6219.016167],
#     'A3': [5.45, 21.67, 1.9787, 0.884333333, 46.7648, 6368.797333, 6413.207833, 6724.5285, 1719.62715, 6330.912333],
#     'A5': [0, 0, 0.372, 0.372, 53.43266667, 6477.996727, 6515.485455, 6831.564364, 1775.036182, 6441.127455],
#     'A7': [0, 0, 0.085102041, 0.1012, 3.4548, 6677.367167, 6755.188667, 7627.668333, 1637.069333, 6612.9425],
#     'A9': [0, 0, 0, 0, 0, 6886.4908, 6946.7564, 7666.3768, 1779.989, 6834.5838]
# }
# row_names = ['发芽势', '发芽率', '根长', '胚根鞘长', '平均O2-浓度', '714.6', '723.2', '871.2', '432.1', '708.2']
# df = df = pd.DataFrame(data, index=row_names)

# ===================== 2. 数据检查与清理 =====================
print("=== 数据基本信息 ===")
print(f"数据形状：{df.shape} (行：指标/波长 | 列：A0/A1/A3/A5/A7/A9)")
print(f"列名：{df.columns.tolist()}")
print(f"行名：{df.index.tolist()}")

# 检查数值类型（确保无字符串）
df = df.astype(float)  # 强制转为浮点数
# 填充空值（如果有）
df = df.fillna(df.mean())

# ===================== 3. 计算相关性矩阵 =====================
# 计算列之间的相关性（A0/A1/A3/A5/A7/A9之间的相关性）
# 若想计算行之间的相关性，改为：corr = df.T.corr(method="pearson")
corr = df.T.corr(method="pearson")  # 可选：pearson(线性)/spearman(秩相关)

print("\n=== 相关性矩阵（A0-A9之间）===")
print(corr.round(2))

# ===================== 4. 绘制专业级相关性热图 =====================
# 核心修改：设置全局字体为Times New Roman（新罗马字体）
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']  # 优先新罗马，兼容中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体设为新罗马

plt.figure(figsize=(10, 8))  # 画布尺寸适配6列数据
ax = sns.heatmap(
    corr,
    annot=True,               # 显示相关系数数值
    fmt=".2f",                # 保留两位小数
    cmap="RdYlBu_r",          # 红蓝配色（更适合科研图表）
    vmin=-1, vmax=1,          # 颜色范围固定在-1到1
    square=True,              # 单元格为正方形
    cbar_kws={
        "shrink": 0.8,
        "label": "Pearson相关系数"
    },                        # 颜色条配置
    linewidths=0.8,           # 单元格间分隔线（更清晰）
    annot_kws={
        "size": 11,
        "fontfamily": 'Times New Roman'  # 标注文字强制设为新罗马
    },   # 标注文字大小
    cbar=True                 # 显示颜色条
)

# 优化标题和标签（字体自动继承全局的Times New Roman）
plt.title("A0/A1/A3/A5/A7/A9 相关性热图", fontsize=16, pad=20)
plt.xticks(rotation=0, fontsize=12)  # X轴标签不旋转（6列无需旋转）
plt.yticks(rotation=0, fontsize=12)  # Y轴标签不旋转

# 调整布局，避免标签截断
plt.tight_layout()

# 保存高清图片（可直接用于论文/报告）
plt.savefig(
    "A0-A9相关性热图6.png",
    dpi=300,        # 高清分辨率
    bbox_inches="tight",  # 去除多余空白
    facecolor="white"     # 背景为白色（默认透明）
)

# 显示热图
plt.show()