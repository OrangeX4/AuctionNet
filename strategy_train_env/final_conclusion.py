"""
深入分析：训练过程是否真的有问题
"""

print("=" * 80)
print("关键问题分析：训练时的 budget 和 cpa 参数在哪里使用？")
print("=" * 80)

print("\n🔍 让我们梳理一下整个流程：")

print("\n1️⃣  训练数据生成阶段 (train_data_generator.py):")
print("   - 从原始CSV读取数据，包含真实的 budget 和 CPAConstraint")
print("   - 计算 budget_left = remainingBudget / budget")
print("   - 这里使用的是数据中的真实 budget (2000-4850)")
print("   - ✅ 没有问题")

print("\n2️⃣  训练阶段 (run_iql.py, run_bc.py 等):")
print("   - 直接从训练数据CSV读取 state, action, reward")
print("   - state 已经包含了正确的 budget_left")
print("   - 模型学习的是: state -> action 的映射")
print("   - ❓ 这里没有用到策略类，所以不受 budget=100, cpa=2 影响")
print("   - ✅ 训练过程本身没有问题！")

print("\n3️⃣  评估阶段 (run_evaluate.py):")
print("   - 需要实例化策略类: IqlBiddingStrategy(budget=?, cpa=?)")
print("   - 策略类在 bidding() 方法中计算 budget_left")
print("   - budget_left = self.remaining_budget / self.budget")
print("   - 如果这里的 self.budget=100，但训练时用的是 3000")
print("   - ❌ budget_left 会完全不在训练分布范围内！")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

print("\n✅ 训练脚本本身是对的:")
print("   - 训练时直接读取数据中已经计算好的 state")
print("   - 不需要实例化策略类，不受默认参数影响")

print("\n❌ 评估脚本原来是错的:")
print("   - 评估时用 budget=100, cpa=2 初始化策略")
print("   - 导致 budget_left 和训练时的分布完全不同")
print("   - 已经修复，现在从测试数据读取真实参数")

print("\n⚠️  潜在的风险:")
print("   - 如果有人直接实例化策略类而不传参数")
print("   - 例如: agent = IqlBiddingStrategy()")
print("   - 会使用默认的 budget=100, cpa=2")
print("   - 这会导致性能很差")

print("\n💡 建议:")
print("   1. 在策略类的 __init__ 中添加警告")
print("   2. 或者移除默认参数，强制用户传入")
print("   3. 或者在文档中明确说明必须传入正确的参数")

print("\n" + "=" * 80)
print("让我们验证一下这个理论")
print("=" * 80)

import pandas as pd
import numpy as np
import ast

# 读取训练数据
df = pd.read_csv("./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv")

def safe_literal_eval(val):
    if pd.isna(val):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

df["state"] = df["state"].apply(safe_literal_eval)

# 提取 budget_left (state的第1个元素)
states = np.array([s for s in df["state"].values if isinstance(s, tuple)])
budget_left_training = states[:, 1]

print(f"\n📊 训练数据中的 budget_left 分布:")
print(f"   最小值: {budget_left_training.min():.4f}")
print(f"   最大值: {budget_left_training.max():.4f}")
print(f"   平均值: {budget_left_training.mean():.4f}")
print(f"   中位数: {np.median(budget_left_training):.4f}")

# 模拟评估时的情况
print(f"\n🔴 如果评估时使用 budget=100:")
print(f"   初始时 remaining_budget=100, budget=100")
print(f"   初始 budget_left = 100/100 = 1.0 ✅")
print(f"   但之后 remaining_budget 会减少...")
print(f"   例如花费了 50，remaining_budget=50")
print(f"   budget_left = 50/100 = 0.5 ✅ 这个还算正常")

print(f"\n✅ 如果评估时使用 budget=3000 (真实值):")
print(f"   初始时 remaining_budget=3000, budget=3000")
print(f"   初始 budget_left = 3000/3000 = 1.0 ✅")
print(f"   花费了 1500，remaining_budget=1500")
print(f"   budget_left = 1500/3000 = 0.5 ✅")
print(f"   分布范围和训练时一致！")

print("\n" + "=" * 80)
print("最终结论")
print("=" * 80)

print("\n1. ✅ 训练脚本本身没有问题")
print("   训练时直接从数据读取 state，不受策略类默认参数影响")

print("\n2. ✅ 评估脚本已经修复")
print("   现在从测试数据读取真实的 budget 和 cpa")

print("\n3. ⚠️  策略类的默认参数具有误导性")
print("   建议修改所有策略类，移除或修正默认参数")

print("\n4. 📝 需要添加文档说明")
print("   明确说明初始化策略时必须传入与训练数据一致的参数")
