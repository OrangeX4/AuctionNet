import numpy as np
import math
import logging
import argparse
import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auctionbid.offline_eval.test_dataloader import TestDataLoader
from auctionbid.offline_eval.offline_env import OfflineEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_strategy_class(algorithm):
    """
    Dynamically import the bidding strategy class based on the algorithm name.
    """
    strategy_map = {
        'iql': ('auctionbid.strategy.iql_bidding_strategy', 'IqlBiddingStrategy'),
        'bc': ('auctionbid.strategy.bc_bidding_strategy', 'BcBiddingStrategy'),
        'bcq': ('auctionbid.strategy.bcq_bidding_strategy', 'BcqBiddingStrategy'),
        'cql': ('auctionbid.strategy.cql_bidding_strategy', 'CqlBiddingStrategy'),
        'td3_bc': ('auctionbid.strategy.td3_bc_bidding_strategy', 'TD3_BCBiddingStrategy'),
        'onlinelp': ('auctionbid.strategy.onlinelp_bidding_strategy', 'OnlineLpBiddingStrategy'),
        'dt': ('auctionbid.strategy.dt_bidding_strategy', 'DtBiddingStrategy'),
    }
    if algorithm not in strategy_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {list(strategy_map.keys())}")
    
    module_name, class_name = strategy_map[algorithm]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test(algorithm):
    """
    offline evaluation
    """
    PlayerBiddingStrategy = get_strategy_class(algorithm)

    data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
    env = OfflineEnv()

    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]

    # 从数据中读取正确的预算和CPA约束
    test_data = test_dict[key]
    actual_budget = test_data['budget'].iloc[0]
    actual_cpa = test_data['CPAConstraint'].iloc[0]

    logger.info(f'Testing advertiser: deliveryPeriod={key[0]}, advertiser={key[1]}')
    logger.info(f'Actual budget from data: {actual_budget}')
    logger.info(f'Actual CPA constraint from data: {actual_cpa}')

    # 使用数据中的真实参数初始化策略
    agent = PlayerBiddingStrategy(budget=actual_budget, cpa=actual_cpa)
    print(agent.name)

    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
    rewards = np.zeros(num_timeStepIndex)
    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }

    for timeStep_index in range(num_timeStepIndex):
        logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

        pValue = pValues[timeStep_index]
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:

            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])

        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)
        logger.info(f'Timestep Index: {timeStep_index + 1} End')
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    score = getScore_neurips(all_reward, cpa_real, cpa_constraint)

    logger.info(f'Total Reward: {all_reward}')
    logger.info(f'Total Cost: {all_cost}')
    logger.info(f'CPA-real: {cpa_real}')
    logger.info(f'CPA-constraint: {cpa_constraint}')
    logger.info(f'Score: {score}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run offline evaluation for bidding strategies.')
    parser.add_argument('--algo', type=str, default='iql', 
                        choices=['iql', 'bc', 'bcq', 'cql', 'td3_bc', 'onlinelp', 'dt'],
                        help='The bidding algorithm to evaluate.')
    args = parser.parse_args()
    run_test(args.algo)
