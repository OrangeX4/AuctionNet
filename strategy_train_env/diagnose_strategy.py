import numpy as np
import torch
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

print("=" * 80)
print("诊断脚本 - 分析策略性能问题")
print("=" * 80)

# 1. 检查数据加载
print("\n[1] 检查测试数据...")
data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
keys = data_loader.keys
print(f"✓ 找到 {len(keys)} 个测试案例")
print(f"✓ 第一个key: {keys[0]}")

key = keys[0]
num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
print(f"✓ 时间步数: {num_timeStepIndex}")
print(f"✓ 第一个时间步的PV数量: {len(pValues[0])}")
print(f"✓ pValues范围: [{pValues[0].min():.4f}, {pValues[0].max():.4f}]")
print(f"✓ leastWinningCosts范围: [{leastWinningCosts[0].min():.4f}, {leastWinningCosts[0].max():.4f}]")

# 2. 检查策略初始化
print("\n[2] 检查策略初始化...")
agent = PlayerBiddingStrategy()
print(f"✓ 策略名称: {agent.name}")
print(f"✓ 初始预算: {agent.budget}")
print(f"✓ CPA约束: {agent.cpa}")
print(f"✓ 剩余预算: {agent.remaining_budget}")

# 3. 检查第一次出价
print("\n[3] 测试第一次出价...")
history = {
    'historyBids': [],
    'historyAuctionResult': [],
    'historyImpressionResult': [],
    'historyLeastWinningCost': [],
    'historyPValueInfo': []
}

timeStep_index = 0
pValue = pValues[timeStep_index]
pValueSigma = pValueSigmas[timeStep_index]
leastWinningCost = leastWinningCosts[timeStep_index]

print(f"✓ pValue样本 (前5个): {pValue[:5]}")
print(f"✓ pValueSigma样本 (前5个): {pValueSigma[:5]}")
print(f"✓ leastWinningCost样本 (前5个): {leastWinningCost[:5]}")

bid = agent.bidding(timeStep_index, pValue, pValueSigma, 
                   history["historyPValueInfo"],
                   history["historyBids"],
                   history["historyAuctionResult"], 
                   history["historyImpressionResult"],
                   history["historyLeastWinningCost"])

print(f"\n出价结果:")
print(f"  - 出价形状: {bid.shape}")
print(f"  - 出价范围: [{bid.min():.6f}, {bid.max():.6f}]")
print(f"  - 出价均值: {bid.mean():.6f}")
print(f"  - 出价中位数: {np.median(bid):.6f}")
print(f"  - 非零出价数量: {np.sum(bid > 0)}")
print(f"  - 出价样本 (前10个): {bid[:10]}")

# 4. 检查拍卖结果
print("\n[4] 模拟拍卖...")
env = OfflineEnv()
tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
    pValue, pValueSigma, bid, leastWinningCost)

print(f"拍卖结果:")
print(f"  - 赢得的拍卖数量: {np.sum(tick_status)}")
print(f"  - 总成本: {np.sum(tick_cost):.2f}")
print(f"  - 总转化: {np.sum(tick_conversion)}")
print(f"  - tick_status样本: {tick_status[:10]}")
print(f"  - tick_cost样本: {tick_cost[:10]}")
print(f"  - tick_conversion样本: {tick_conversion[:10]}")

# 5. 分析出价与市场价格的关系
print("\n[5] 分析出价策略...")
win_rate = np.sum(bid >= leastWinningCost) / len(bid)
print(f"  - 理论胜率 (bid >= leastWinningCost): {win_rate:.2%}")
print(f"  - 出价/市场价格比率 (均值): {(bid.mean() / leastWinningCost.mean()):.4f}")

# 比较出价和市场价格的分布
print(f"\n  出价 vs 市场价格对比 (前20个):")
for i in range(min(20, len(bid))):
    status = "✓赢" if bid[i] >= leastWinningCost[i] else "✗输"
    print(f"    [{i:2d}] 出价={bid[i]:8.6f}, 市场价={leastWinningCost[i]:8.6f}, pValue={pValue[i]:8.6f} {status}")

# 6. 检查模型加载
print("\n[6] 检查模型细节...")
try:
    if hasattr(agent, 'model'):
        print(f"✓ 模型类型: {type(agent.model)}")
        
        # 尝试获取模型参数信息
        if hasattr(agent.model, 'parameters'):
            try:
                params = list(agent.model.parameters())
                total_params = sum(p.numel() for p in params)
                print(f"✓ 模型参数总数: {total_params}")
            except:
                print("  (无法获取参数信息 - 可能是JIT模型)")
        
    if hasattr(agent, 'normalize_dict'):
        print(f"✓ 归一化字典键: {list(agent.normalize_dict.keys())}")
        print(f"✓ 归一化字典示例:")
        for i, (k, v) in enumerate(list(agent.normalize_dict.items())[:3]):
            print(f"    键{k}: min={v['min']:.6f}, max={v['max']:.6f}")
except Exception as e:
    print(f"✗ 检查模型时出错: {e}")

# 7. 测试多个时间步
print("\n[7] 运行完整测试 (前10个时间步)...")
agent.reset()
history = {
    'historyBids': [],
    'historyAuctionResult': [],
    'historyImpressionResult': [],
    'historyLeastWinningCost': [],
    'historyPValueInfo': []
}

total_conversions = 0
total_cost = 0

for t in range(min(10, num_timeStepIndex)):
    pValue = pValues[t]
    pValueSigma = pValueSigmas[t]
    leastWinningCost = leastWinningCosts[t]
    
    bid = agent.bidding(t, pValue, pValueSigma,
                       history["historyPValueInfo"],
                       history["historyBids"],
                       history["historyAuctionResult"],
                       history["historyImpressionResult"],
                       history["historyLeastWinningCost"])
    
    tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
        pValue, pValueSigma, bid, leastWinningCost)
    
    step_cost = np.sum(tick_cost)
    step_conversion = np.sum(tick_conversion)
    
    agent.remaining_budget -= step_cost
    total_conversions += step_conversion
    total_cost += step_cost
    
    # 更新历史
    temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
    history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
    history["historyBids"].append(bid)
    history["historyLeastWinningCost"].append(leastWinningCost)
    temAuctionResult = np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
    history["historyAuctionResult"].append(temAuctionResult)
    temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
    history["historyImpressionResult"].append(temImpressionResult)
    
    print(f"  步骤{t+1}: 赢得={np.sum(tick_status):4d}, 转化={step_conversion:3.0f}, 成本={step_cost:6.2f}, 剩余预算={agent.remaining_budget:6.2f}")

print(f"\n前10步总结:")
print(f"  - 总转化: {total_conversions}")
print(f"  - 总成本: {total_cost:.2f}")
print(f"  - 实际CPA: {total_cost/(total_conversions+1e-10):.2f}")
print(f"  - 目标CPA: {agent.cpa}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
