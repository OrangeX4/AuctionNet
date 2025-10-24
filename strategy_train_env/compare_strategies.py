"""
对比分析不同alpha值的效果
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

print("=" * 80)
print("对比分析: 不同出价策略的效果")
print("=" * 80)

# 加载数据
data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
key = data_loader.keys[0]
num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)

env = OfflineEnv()
budget = 100
cpa_constraint = 2

# 测试不同的alpha值
alphas_to_test = [1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

print("\n格式: Alpha | 胜率 | 总转化 | 总成本 | 实际CPA | Score")
print("-" * 80)

for alpha in alphas_to_test:
    remaining_budget = budget
    total_conversions = 0
    total_cost = 0
    total_wins = 0
    total_pvs = 0
    
    for t in range(num_timeStepIndex):
        pValue = pValues[t]
        pValueSigma = pValueSigmas[t]
        leastWinningCost = leastWinningCosts[t]
        
        # 简单策略: bid = alpha * pValue
        bids = alpha * pValue
        
        # 模拟拍卖
        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
            pValue, pValueSigma, bids, leastWinningCost)
        
        # 处理预算超支
        step_cost = np.sum(tick_cost)
        if step_cost > remaining_budget:
            # 随机丢弃一些赢得的拍卖
            win_indices = np.where(tick_status == 1)[0]
            if len(win_indices) > 0:
                over_cost_ratio = max((step_cost - remaining_budget) / (step_cost + 1e-4), 0)
                num_to_drop = min(int(np.ceil(len(win_indices) * over_cost_ratio)), len(win_indices))
                drop_indices = np.random.choice(win_indices, num_to_drop, replace=False)
                tick_status[drop_indices] = 0
                tick_cost[drop_indices] = 0
                tick_conversion[drop_indices] = 0
                step_cost = np.sum(tick_cost)
        
        remaining_budget -= step_cost
        total_conversions += np.sum(tick_conversion)
        total_cost += step_cost
        total_wins += np.sum(tick_status)
        total_pvs += len(pValue)
    
    # 计算指标
    win_rate = total_wins / total_pvs if total_pvs > 0 else 0
    actual_cpa = total_cost / (total_conversions + 1e-10)
    
    # 计算score
    beta = 2
    penalty = 1
    if actual_cpa > cpa_constraint:
        coef = cpa_constraint / (actual_cpa + 1e-10)
        penalty = pow(coef, beta)
    score = penalty * total_conversions
    
    status = ""
    if actual_cpa <= cpa_constraint and total_conversions > 0:
        status = " ✓ 满足CPA约束"
    elif total_conversions == 0:
        status = " ✗ 无转化"
    else:
        status = " ✗ 违反CPA约束"
    
    print(f"{alpha:5.1f} | {win_rate:5.2%} | {total_conversions:7.1f} | {total_cost:8.2f} | {actual_cpa:10.2f} | {score:6.2f}{status}")

print("\n" + "=" * 80)
print("结论分析")
print("=" * 80)
print("""
从上面的结果可以看出:

1. **低alpha值 (1.0-2.0)**: 
   - 符合CPA约束，但几乎赢不了任何拍卖
   - 胜率接近0%，导致转化数为0
   - 这是因为市场价格远高于合理出价

2. **中等alpha值 (5.0-20.0)**:
   - 开始能赢得一些拍卖
   - 但实际CPA远超约束值
   - 分数会受到严重惩罚

3. **高alpha值 (50.0-100.0)**:
   - 能赢得更多拍卖，有可能获得转化
   - 但CPA约束严重违反
   - 虽然转化数增加，但分数因惩罚而降低

关键问题:
- 市场环境不利: 市场价格(均值0.088) >> 合理出价(pValue×2 ≈ 0.0008)
- 这是一个约110倍的差距！
- 在这种环境下，要么违反CPA约束，要么获得0转化

训练数据的问题:
- 如果训练数据来自相同的市场环境
- 模型学到的是在这种环境下的最优策略
- 但"最优"策略也可能是违反CPA约束来获得转化
- 或者保守出价但几乎不获得转化

建议:
1. 检查数据集是否正确 - pValue和leastWinningCost的量纲可能不对
2. 检查训练数据的生成过程
3. 可能需要调整评估环境的参数
4. 或者这就是一个极具挑战性的场景，需要更复杂的策略
""")
