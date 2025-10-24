"""
检查原始CSV数据的实际值
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("检查原始CSV数据")
print("=" * 80)

# 读取CSV文件的前几行
df = pd.read_csv('./data/traffic/period-7.csv', nrows=100000)

print(f"\n数据形状: {df.shape}")
print(f"\n列名: {list(df.columns)}")

print(f"\n前5行数据:")
print(df.head(10))

print(f"\n数据统计:")
print(df.describe())

print(f"\n关键列的值分布:")
if 'pValue' in df.columns:
    print(f"\npValue:")
    print(f"  范围: [{df['pValue'].min():.6f}, {df['pValue'].max():.6f}]")
    print(f"  均值: {df['pValue'].mean():.6f}")
    print(f"  中位数: {df['pValue'].median():.6f}")
    print(f"  分位数:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    {p}%: {df['pValue'].quantile(p/100):.6f}")

if 'leastWinningCost' in df.columns:
    print(f"\nleastWinningCost:")
    print(f"  范围: [{df['leastWinningCost'].min():.6f}, {df['leastWinningCost'].max():.6f}]")
    print(f"  均值: {df['leastWinningCost'].mean():.6f}")
    print(f"  中位数: {df['leastWinningCost'].median():.6f}")
    print(f"  分位数:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    {p}%: {df['leastWinningCost'].quantile(p/100):.6f}")

if 'pValueSigma' in df.columns:
    print(f"\npValueSigma:")
    print(f"  范围: [{df['pValueSigma'].min():.6f}, {df['pValueSigma'].max():.6f}]")
    print(f"  均值: {df['pValueSigma'].mean():.6f}")

print(f"\n按timeStepIndex分组统计:")
if 'timeStepIndex' in df.columns:
    grouped = df.groupby('timeStepIndex').size()
    print(f"  时间步数量: {len(grouped)}")
    print(f"  每个时间步的PV数量范围: [{grouped.min()}, {grouped.max()}]")
    print(f"  每个时间步的PV数量均值: {grouped.mean():.0f}")

# 检查是否有其他可能相关的列
print(f"\n所有列的数据类型:")
print(df.dtypes)

# 检查pValue和leastWinningCost的关系
if 'pValue' in df.columns and 'leastWinningCost' in df.columns:
    print(f"\npValue vs leastWinningCost:")
    print(f"  leastWinningCost / pValue 比率:")
    ratio = df['leastWinningCost'] / (df['pValue'] + 1e-10)
    print(f"    均值: {ratio.mean():.2f}")
    print(f"    中位数: {ratio.median():.2f}")
    print(f"  这意味着市场价格是pValue的{ratio.median():.0f}倍")
    
    # 如果按CPA=2出价，能赢得多少？
    optimal_bid = df['pValue'] * 2
    can_win = optimal_bid >= df['leastWinningCost']
    print(f"\n  如果出价 = pValue × 2:")
    print(f"    能赢得的比例: {can_win.mean()*100:.4f}%")
    print(f"    能赢得的数量: {can_win.sum()} / {len(can_win)}")
