import numpy as np
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class SimpleBiddingStrategy(BaseBiddingStrategy):
    """
    一个简单但有效的出价策略
    核心思想: bid = alpha * pValue, 其中alpha根据剩余预算和时间动态调整
    """

    def __init__(self, budget=100, name="Simple-Strategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        self.base_alpha = 1.5  # 基础出价系数，保守设置为CPA的75%

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        简单但合理的出价策略
        
        策略逻辑:
        1. bid = alpha * pValue，确保期望CPA不超过约束
        2. alpha根据剩余预算和时间动态调整
        3. 预算充足时可以略微激进，预算紧张时更保守
        """
        
        # 计算进度
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        
        # 动态调整alpha
        # 如果预算剩余比时间剩余多，可以更激进一些
        # 如果预算紧张，需要更保守
        if budget_left > time_left * 1.2:
            # 预算充足，可以略微激进
            alpha = self.base_alpha * 1.2
        elif budget_left < time_left * 0.8:
            # 预算紧张，更保守
            alpha = self.base_alpha * 0.8
        else:
            # 正常情况
            alpha = self.base_alpha
        
        # 确保alpha不超过CPA约束
        alpha = min(alpha, self.cpa * 0.9)  # 留10%安全边际
        
        # 计算出价
        bids = alpha * pValues
        
        # 确保出价非负
        bids = np.maximum(bids, 0)
        
        return bids
