import time
import gin
import numpy as np
import os
import psutil
# from saved_model.DTtest.dt import DecisionTransformer
from bidding_train_env.baseline.qt.ql_DT import DecisionTransformer, Critic
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle


class DtBiddingStrategy(BaseBiddingStrategy):
    """
    Decision-Transformer-PlayerStrategy
    """

    def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "QTtest", "dt.pt")
        picklePath = os.path.join(dir_name, "saved_model", "QTtest", "normalize_dict.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        self.model =    DecisionTransformer(
                            state_dim=16,
                            act_dim=1,
                            max_length=20,
                            max_ep_len=96,
                            hidden_size=256,#variant['embed_dim']
                            n_layer=4,
                            n_head=4,
                            n_inner=4 * 256,
                            activation_function='relu',
                            n_positions=1024,
                            resid_pdrop=0.1,
                            attn_pdrop=0.1,
                            scale=2000,
                            sar=False,
                            rtg_no_q=False,
                            infer_no_q=False
                        )
        self.model.load_net(model_path)
        self.model.to(device)
        self.model.state_mean = normalize_dict['state_mean']
        self.model.state_std = normalize_dict['state_std']

        model_path_critic = os.path.join(dir_name, "saved_model", "QTtest", "critic.pt")

        self.critic = Critic(state_dim=16, action_dim=1)
        self.critic.load_state_dict(torch.load(model_path_critic, map_location=device))
        self.critic.to(device)


    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()

        alpha = self.model.take_actions(self.critic, test_state,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None)
        bids = alpha * pValues * 200
        return bids[0]


