import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bidding_train_env.offline_eval.test_dataloader import TestDataLoader

print("=" * 80)
print("深度分析: 为什么转化率为0?")
print("=" * 80)

# 加载数据
data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
key = data_loader.keys[0]
num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)

print("\n[关键发现] pValue (转化概率) 的分布:")
all_pvalues = np.concatenate(pValues)
print(f"  - 总PV数量: {len(all_pvalues)}")
print(f"  - pValue范围: [{all_pvalues.min():.6f}, {all_pvalues.max():.6f}]")
print(f"  - pValue均值: {all_pvalues.mean():.6f}")
print(f"  - pValue中位数: {np.median(all_pvalues):.6f}")
print(f"  - pValue标准差: {all_pvalues.std():.6f}")

print("\n  pValue分布统计:")
percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    val = np.percentile(all_pvalues, p)
    print(f"    {p:3d}%分位数: {val:.6f}")

print("\n[问题分析]")
print(f"  ⚠ pValue均值仅为 {all_pvalues.mean():.6f} ({all_pvalues.mean()*100:.4f}%)")
print(f"  ⚠ 这意味着即使赢得拍卖，每个PV的期望转化率也极低")
print(f"  ⚠ 例如: 赢得100个拍卖，期望转化数 = 100 × {all_pvalues.mean():.6f} = {100*all_pvalues.mean():.3f}")

print("\n[市场价格分析]")
all_costs = np.concatenate(leastWinningCosts)
print(f"  - leastWinningCost范围: [{all_costs.min():.6f}, {all_costs.max():.6f}]")
print(f"  - leastWinningCost均值: {all_costs.mean():.6f}")
print(f"  - leastWinningCost中位数: {np.median(all_costs):.6f}")

print("\n[ROI分析]")
print(f"  如果出价等于市场价格的平均值 {all_costs.mean():.4f}:")
print(f"    - 每次赢得拍卖的期望转化: {all_pvalues.mean():.6f}")
print(f"    - 每次转化的期望成本: {all_costs.mean()/all_pvalues.mean():.2f}")
print(f"    - 而CPA约束是: 2.00")
print(f"    ⚠ 这意味着市场价格远高于合理的出价!")

print("\n[最优策略分析]")
print(f"  要满足CPA≤2的约束，出价应该是:")
print(f"    bid ≤ pValue × CPA")
print(f"    bid ≤ pValue × 2")
print(f"  ")
print(f"  对于pValue={all_pvalues.mean():.6f}的情况:")
print(f"    最大出价 = {all_pvalues.mean() * 2:.6f}")
print(f"    而市场价格均值 = {all_costs.mean():.6f}")
print(f"    ⚠ 最大出价是市场价格的 {(all_pvalues.mean() * 2 / all_costs.mean())*100:.2f}%")

print("\n[胜率计算]")
optimal_bids = all_pvalues * 2  # CPA=2的最优出价
win_rate = np.mean(optimal_bids >= all_costs)
print(f"  如果按照 bid = pValue × 2 出价:")
print(f"    理论胜率: {win_rate*100:.2f}%")
print(f"    在{len(all_pvalues)}个PV中，能赢得约 {int(len(all_pvalues)*win_rate)} 个")

# 计算期望收益
winning_indices = optimal_bids >= all_costs
expected_cost = np.sum(all_costs[winning_indices])
expected_conversions = np.sum(all_pvalues[winning_indices])
print(f"\n  期望结果:")
print(f"    总成本: {expected_cost:.2f}")
print(f"    期望转化数: {expected_conversions:.2f}")
print(f"    实际CPA: {expected_cost/(expected_conversions+1e-10):.2f}")

print("\n[策略出价分析]")
print(f"  当前策略出价 = alpha × pValue")
print(f"  从诊断结果看，alpha ≈ {0.039202/all_pvalues.mean():.2f}")
print(f"  这个alpha值意味着策略正在出价 ≈ {0.039202/all_pvalues.mean():.2f} × pValue")
print(f"  ")
print(f"  理论上，为了满足CPA=2:")
print(f"    应该: alpha × pValue × (1/pValue) ≤ CPA")
print(f"    即: alpha ≤ CPA = 2")
print(f"  ")
print(f"  但考虑到转化的随机性和市场竞争:")
print(f"    alpha通常应该在 CPA/2 到 CPA 之间")
print(f"    即: 1.0 到 2.0 之间")
print(f"  ")
print(f"  ⚠ 当前的alpha={0.039202/all_pvalues.mean():.2f} 远高于这个范围!")

print("\n" + "=" * 80)
print("根本原因总结:")
print("=" * 80)
print("""
1. 【数据特征】
   - pValue (转化概率) 极低，平均只有 0.0006 (0.06%)
   - 市场价格 (leastWinningCost) 相对较高，平均 0.088
   - 这是一个极低转化率的广告场景

2. 【市场环境】
   - 在这种低转化率环境下，市场价格 >> 合理出价
   - 要满足CPA=2约束，出价应该 ≈ pValue × 2 ≈ 0.0012
   - 但市场价格均值是 0.088，约为合理出价的 73倍！

3. 【策略问题】
   - 当前策略的alpha值过高 (≈64)，导致出价过高
   - 出价 ≈ 0.039，远高于合理的 0.0012
   - 虽然提高了胜率(4.69%)，但赢得的拍卖都是"亏本生意"
   - 由于转化概率极低，即使赢得拍卖也几乎不产生转化

4. 【为什么转化为0】
   - 转化 ~ Binomial(n=1, p=pValue)
   - 当pValue≈0.0006时，单次转化概率只有0.06%
   - 需要赢得大约1667次拍卖才能期望获得1次转化
   - 但赢得这么多拍卖会耗尽预算且严重违反CPA约束

5. 【训练数据的影响】
   - 如果训练数据中也是这种低转化率场景
   - 模型可能学到了"高出价以获得更多转化"的策略
   - 但这会导致CPA约束严重违反
   - 需要更强的CPA约束项或奖励整形

建议解决方案:
1. 降低模型的alpha输出范围，使其接近1-2
2. 在训练时加入更强的CPA约束惩罚
3. 使用更保守的出价策略: bid = min(pValue * 1.5, market_price)
4. 考虑实现一个简单的基线策略来验证环境是否合理
""")
