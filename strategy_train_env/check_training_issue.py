"""
检查训练脚本中的参数设置问题
"""
import pandas as pd
import numpy as np
import ast

def check_training_data_params():
    """检查训练数据中的参数"""
    print("=" * 80)
    print("检查训练数据参数")
    print("=" * 80)
    
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    df = pd.read_csv(train_data_path)
    
    print(f"\n训练数据总行数: {len(df)}")
    print(f"\n唯一的 Budget 值: {sorted(df['budget'].unique())}")
    print(f"Budget 范围: {df['budget'].min()} - {df['budget'].max()}")
    print(f"\n唯一的 CPAConstraint 值: {sorted(df['CPAConstraint'].unique())}")
    print(f"CPAConstraint 范围: {df['CPAConstraint'].min()} - {df['CPAConstraint'].max()}")
    
    # 统计每个 budget-cpa 组合的数据量
    print("\n" + "=" * 80)
    print("Budget-CPA 组合统计")
    print("=" * 80)
    group_stats = df.groupby(['budget', 'CPAConstraint']).size().reset_index(name='count')
    print(group_stats.to_string(index=False))
    
    return df

def check_state_features(df):
    """检查状态特征的分布"""
    print("\n" + "=" * 80)
    print("检查状态特征分布")
    print("=" * 80)
    
    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    
    df["state"] = df["state"].apply(safe_literal_eval)
    
    # 提取所有状态特征
    states = np.array([s for s in df["state"].values if isinstance(s, tuple)])
    
    print(f"\nState 特征维度: {states.shape}")
    print(f"\n各维度统计 (前几个关键特征):")
    print(f"  [0] time_left - min: {states[:, 0].min():.4f}, max: {states[:, 0].max():.4f}, mean: {states[:, 0].mean():.4f}")
    print(f"  [1] budget_left - min: {states[:, 1].min():.4f}, max: {states[:, 1].max():.4f}, mean: {states[:, 1].mean():.4f}")
    print(f"  [2] avg_bid_all - min: {states[:, 2].min():.4f}, max: {states[:, 2].max():.4f}, mean: {states[:, 2].mean():.4f}")
    
    return states

def check_action_distribution(df):
    """检查动作(alpha)的分布"""
    print("\n" + "=" * 80)
    print("检查动作(Alpha)分布")
    print("=" * 80)
    
    print(f"\nAlpha 统计:")
    print(f"  最小值: {df['action'].min():.4f}")
    print(f"  最大值: {df['action'].max():.4f}")
    print(f"  平均值: {df['action'].mean():.4f}")
    print(f"  中位数: {df['action'].median():.4f}")
    print(f"  标准差: {df['action'].std():.4f}")
    
    # 按budget-cpa分组查看alpha分布
    print("\n按 Budget-CPA 分组的 Alpha 统计:")
    alpha_stats = df.groupby(['budget', 'CPAConstraint'])['action'].agg(['mean', 'std', 'min', 'max'])
    print(alpha_stats.to_string())

def analyze_budget_cpa_relationship():
    """分析 Budget 和 CPA 的关系"""
    print("\n" + "=" * 80)
    print("Budget 和 CPA 的关系分析")
    print("=" * 80)
    
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    df = pd.read_csv(train_data_path)
    
    # 计算实际性能
    performance = df.groupby(['budget', 'CPAConstraint', 'advertiserNumber']).agg({
        'realAllCost': 'first',
        'realAllConversion': 'first'
    }).reset_index()
    
    performance['realCPA'] = performance['realAllCost'] / (performance['realAllConversion'] + 1e-8)
    
    print("\n各 Budget-CPA 组合的实际表现:")
    summary = performance.groupby(['budget', 'CPAConstraint']).agg({
        'realAllCost': 'mean',
        'realAllConversion': 'mean',
        'realCPA': 'mean'
    }).reset_index()
    summary.columns = ['Budget', 'CPA约束', '平均成本', '平均转化', '实际CPA']
    print(summary.to_string(index=False))
    
    # 检查是否满足约束
    print("\n约束满足情况:")
    summary['满足约束'] = summary['实际CPA'] <= summary['CPA约束']
    print(f"满足约束的比例: {summary['满足约束'].sum() / len(summary) * 100:.1f}%")

def check_normalization():
    """检查归一化参数"""
    print("\n" + "=" * 80)
    print("检查归一化参数")
    print("=" * 80)
    
    import pickle
    import os
    
    dict_path = "./saved_model/IQLtest/normalize_dict.pkl"
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as file:
            normalize_dict = pickle.load(file)
        
        print("\n归一化字典内容:")
        for key, value in normalize_dict.items():
            print(f"  维度 {key}: min={value['min']:.6f}, max={value['max']:.6f}")
    else:
        print("\n归一化字典文件不存在!")

def main():
    print("\n" + "=" * 80)
    print("训练脚本参数设置问题全面检查")
    print("=" * 80)
    
    # 1. 检查训练数据参数
    df = check_training_data_params()
    
    # 2. 检查状态特征
    states = check_state_features(df)
    
    # 3. 检查动作分布
    check_action_distribution(df)
    
    # 4. 分析 Budget-CPA 关系
    analyze_budget_cpa_relationship()
    
    # 5. 检查归一化参数
    check_normalization()
    
    print("\n" + "=" * 80)
    print("关键发现总结")
    print("=" * 80)
    print("\n⚠️  训练数据使用的参数范围:")
    print(f"   Budget: 2000 - 4850")
    print(f"   CPA: 60 - 130")
    
    print("\n❌ 策略类中硬编码的默认参数:")
    print(f"   Budget: 100")
    print(f"   CPA: 2")
    
    print("\n🔴 这是一个严重的不匹配问题!")
    print("   策略初始化时使用的默认参数与训练数据完全不匹配")
    print("   这会导致 budget_left 等归一化特征完全错误")
    
    print("\n💡 解决方案:")
    print("   方案1: 修改所有策略类，从测试数据中读取真实的 budget 和 cpa")
    print("   方案2: 重新生成训练数据，使用 budget=100, cpa=2")
    print("   方案3: 让模型学习 budget 和 cpa 无关的特征（更难）")
    
    print("\n✅ 推荐: 方案1 (已在 run_evaluate.py 中实现)")
    print("   但需要检查训练过程中是否也存在这个问题")

if __name__ == "__main__":
    main()
