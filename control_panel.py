from engine import MarkowitzPortfolio as mp


SOURCE: str = None
FIELDS: list = ['Adj Close']
START_DATE: str = '2019-5-4'
END_DATE: str = '2020-5-4'
VOLA_CUT_OFF: float = 1.0

if __name__ == '__main__':
    marko = mp(source=SOURCE, fields=FIELDS, start_date=START_DATE, end_date=END_DATE, vola_cut_off=VOLA_CUT_OFF)
    marko.set_settings(num_of_portfolios=10000)
    marko.run_portfolio_optimization()
    marko.display_simulated_ef_with_random()
