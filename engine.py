# imports ==============================================================================================================

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
from pandas_datareader import data, wb
import os
import bs4 as bs
import pickle
import requests

# imports ==============================================================================================================


class DataHandler:

    def __init__(self, pull_tickers: bool = False) -> None:
        self.project_path: str = os.path.abspath(os.getcwd())
        self.path_dict: dict = {'stock_data': os.path.join(self.project_path, 'price_data.csv'),
                                'tickers': os.path.join(self.project_path, 'tickers.csv')}
        self.local_data_available: bool = os.path.exists(self.path_dict['stock_data'])

        self.tickers: list = list()
        self.number_of_observations: int = 0

        self.raw_stock_data_df: pd.DataFrame = pd.DataFrame()
        self.cleaned_stock_df: pd.DataFrame = pd.DataFrame()
        self.pct_change_returns_df: pd.DataFrame = pd.DataFrame()
        self.mean_returns_df: pd.DataFrame = pd.DataFrame()

        if pull_tickers is True:
            self.pull_sp500_tickers()
        elif pull_tickers is False and os.path.exists(self.path_dict['tickers']):
            self.tickers = pd.read_csv(self.path_dict['tickers'])['tickers'].to_list()
        else:
            self.pull_sp500_tickers()

    def pull_sp500_tickers(self) -> None:
        """
        pulls the s&p 500 symbols from Wikipedia - this method is not called automatically but has to be called manually
        or by initiating this class with pull_tickers=True
        :return: None
        """
        resp: requests = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup: bs.BeautifulSoup = bs.BeautifulSoup(resp.text, 'lxml')
        table: bs.PageElement = soup.find('table', {'class': 'wikitable sortable'})
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.replace('.', '-')
            ticker = ticker[:-1]
            self.tickers.append(ticker)
        with open("sp500tickers.pickle", "wb") as f:
            pickle.dump(self.tickers, f)

    def load_stock_data(self, source: str = None, fields: list = None,
                        start_date: str = None, end_date: str = None) -> None:
        """
        gets stock data as pd.DataFrame into the object from the specified source
        dates only relate to the yahoo download - source=local takes what is available in the files
        :param source: str 'local' or 'yahoo'
        :param fields: list of data fields to receive from yahoo eg. Adj Close - currently it can just process one field
        a list with len == 1
        :param start_date: str date with the format YYYY-MM-DD
        :param end_date: str date with the format YYYY-MM-DD
        :return:
        """
        if (source is None and self.local_data_available) or (source == 'local'):
            self.raw_stock_data_df = pd.read_csv(self.path_dict['stock_data'], index_col=0)
        else:
            self.load_yahoo_data(fields=fields, start_date=start_date, end_date=end_date)

    def load_yahoo_data(self, **kwargs) -> None:
        """
        this method gets stock data from yahoo based on the tickers in the object
        kwargs coming from get_stock_data
        :return: None
        """

        if kwargs['fields'] is None:
            fields = ['Adj Close']
        else:
            fields = kwargs['fields']

        start_date = kwargs['start_date']
        end_date = kwargs['end_date']

        list_of_lists_tickers = [self.tickers[z:z + 5] for z in range(0, len(self.tickers), 5)]
        for l in tqdm(list_of_lists_tickers):
            for item in l:
                stock: pd.DataFrame = data.DataReader(item,
                                                      start=start_date,
                                                      end=end_date,
                                                      data_source='yahoo')[fields]

                self.raw_stock_data_df = pd.concat([self.raw_stock_data_df,
                                                    stock.rename(columns={fields[0]: item})], axis=1, sort=False)
        self.number_of_observations = self.raw_stock_data_df.count()

    def clean_data(self) -> None:
        """
        cleans up the raw data to make them usable for the Markowitz class
        :return: None
        """
        self.cleaned_stock_df = self.raw_stock_data_df.T.drop_duplicates().T
        self.pct_change_returns_df = self.cleaned_stock_df.pct_change().iloc[1:-1].copy()
        self.mean_returns_df = self.pct_change_returns_df.mean()

    def save_data(self) -> None:
        """
        saves the tickers and raw data to make the next process more efficient
        :return:
        """
        pd.DataFrame(self.tickers, columns=['tickers']).to_csv(self.path_dict['tickers'])
        self.raw_stock_data_df.to_csv(self.path_dict['stock_data'])


class MarkowitzPortfolio:

    def __init__(self, source: str, fields: list, start_date: str, end_date: str, vola_cut_off: float) -> None:
        """
        :param source: str
        :param fields: list
        :param start_date: str
        :param end_date: str
        :param vola_cut_off: float - not implemented yet - can be helpful for later boundaries
        """
        self.dh: DataHandler = DataHandler()
        self.dh.load_stock_data(source=source, fields=fields, start_date=start_date, end_date=end_date)
        self.dh.save_data()
        self.dh.clean_data()

        self.mean_returns: pd.DataFrame = self.dh.mean_returns_df
        self.pct_returns: pd.DataFrame = self.dh.pct_change_returns_df

        self.num_of_portfolios: int = 100
        self.num_of_trading_days: int = 252
        self.asset_sample_size: int = 30
        self.risk_free_rate: float = 0.01
        self.vola_cut_off: float = 1.0

        self.random_portfolios: tuple = tuple()

        self.max_sharpe_allocation: np.array = np.array(0)
        self.min_vol_allocation: np.array = np.array(0)

    def load_new_data(self, fields: list, start_date: str, end_date: str) -> None:
        """
        reload new data from yahoo
        :param fields: list of data fields to retrieve - currently just one string eg. "Adj Close"
        :param start_date: str date with format YYYY-MM-DD
        :param end_date: str date with format YYYY-MM-DD
        :return: None
        """
        self.dh.load_stock_data(source='yahoo', fields=fields, start_date=start_date, end_date=end_date)
        self.dh.save_data()
        self.dh.clean_data()

    def set_settings(self, num_of_portfolios: int = 30, num_of_trading_days: int = 252, asset_sample_size: int = 30,
                     risk_free_rate: float = 0.01, vola_cut_off: float = 1.0) -> None:
        self.num_of_portfolios = num_of_portfolios
        self.num_of_trading_days = num_of_trading_days
        self.asset_sample_size = asset_sample_size
        self.risk_free_rate = risk_free_rate
        self.vola_cut_off = vola_cut_off

    def portfolio_annualised_performance(self, weights, returns, cov_matrix):
        returns = np.sum(returns * weights) * self.num_of_trading_days
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(self.num_of_trading_days)
        return std, returns

    def simulate_random_portfolios(self) -> None:
        """
        simulates portfolios based on the set parameters - currently excludes short-selling
        :return: None
        """
        results: np.array = np.zeros((3, self.num_of_portfolios))
        weights_record: list = []
        names_record: list = []

        for i in range(self.num_of_portfolios):
            mean_returns_sample: pd.DataFrame = self.mean_returns.sample(self.asset_sample_size, axis=0)
            sample_assets: list = mean_returns_sample.index.tolist()
            cov_matrix_loop: np.array = self.build_cov(sample_assets=sample_assets)
            weights: np.array = np.random.uniform(0, 1, self.asset_sample_size)
            weights /= np.sum(weights)
            weights_record.append(weights)

            for j in range(self.asset_sample_size):
                if weights[j] < 0:
                    mean_returns_sample[j] = mean_returns_sample[j] * -1
            portfolio_std_dev, portfolio_return = self.portfolio_annualised_performance(weights=weights,
                                                                                        returns=mean_returns_sample,
                                                                                        cov_matrix=cov_matrix_loop)
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - self.risk_free_rate) / portfolio_std_dev
            names_record.append(sample_assets)
        self.random_portfolios = results, weights_record, names_record

    def build_cov(self, sample_assets) -> np.array:
        return np.cov(self.pct_returns[sample_assets].T)

    def run_portfolio_optimization(self) -> None:
        self.simulate_random_portfolios()

    def display_simulated_ef_with_random(self) -> None:
        results, weights, names_record = self.random_portfolios

        results: np.array = results[:, ~np.isnan(results).any(axis=0)]
        max_sharpe_idx: np.array = np.argmax(results[2])
        sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
        max_sharpe_allocation: pd.DataFrame = pd.DataFrame(weights[max_sharpe_idx],
                                                           index=names_record[max_sharpe_idx],
                                                           columns=['allocation'])
        max_sharpe_allocation.loc[:, 'allocation'] = [round(i * 1, 2) for i in max_sharpe_allocation['allocation']]
        self.max_sharpe_allocation = max_sharpe_allocation.T.copy()

        min_vol_idx: np.array = np.argmin(results[0])
        sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
        min_vol_allocation: pd.DataFrame = pd.DataFrame(weights[min_vol_idx],
                                                        index=names_record[min_vol_idx],
                                                        columns=['allocation'])
        min_vol_allocation.loc[:, 'allocation'] = [round(i * 1, 2)for i in min_vol_allocation['allocation']]
        self.min_vol_allocation = min_vol_allocation.T.copy()

        plt.figure(figsize=(10, 7))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
        plt.scatter(sdp_min, rp_min, marker='*', color='y', s=500, label='Minimum volatility')
        plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
        plt.xlabel('annualised volatility')
        plt.ylabel('annualised returns')
        plt.legend(labelspacing=0.8)



# ## Next Steps
# - Constraint on max position size (and minimum size)
# - Possibility of short sales
# - Optionality of cash
#
# - Import xls my positions from xls file and return individual 10 day, 30 day and annual volatility, and sharpe ratio
# - Display my portfolio on the Efficient Frontier
#
# Next level:
# - No fractional shares
# - Maximum number of positions
#
# If short sale, sign of return needs to be changed


