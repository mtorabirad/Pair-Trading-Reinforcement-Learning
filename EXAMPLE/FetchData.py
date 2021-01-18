import time
import sys
sys.path.insert(0, r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code')
from UTIL import FileIO
from DATA import API


config_path = 'CONFIG\config_data.yml'
config      = FileIO.read_yaml(config_path)
attr        = ['date', 'close']
# https://dailypik.com/top-50-companies-sp-500/

#ticker_list = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'FB', 'GOOG']
#ticker_list = ['NVDA', 'CRM', 'PYPL', 'NFLX', 'ABT', 'TMO', 'ABBV', 'COST']
# ticker_list = ['QCOM', 'NKE', 'MCD', 'ACN', 'MDT', 'AVGO', 'NEE', 'DHR', 'BMY', 'TXN', 'HON', 'AMGN', 'LIN']
#ticker_list = ['BRK-B', 'V', 'JPM', 'JNJ', 'WMT']
#ticker_list = ['MA', 'PG', 'BAC', 'INTC', 'T']

#ticker_list = ['UNH', 'XOM', 'HD', 'DIS','KO']
#ticker_list = ['VZ', 'MRK', 'PFE', 'CVX', 'CSCO']
#ticker_list = ['CMCSA', 'PEP', 'WFC', 'BA', 'ADBE']
import pickle
def load_dic(name ):
    with open(r'DATA/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

sec_sym_dic = load_dic('sec_sym_dic')

#curr_sector = 'Information Technology'
#curr_sector = 'Health Care'
curr_sector = 'Financials'

import os
directory = r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code\STATICS\S&P500\\' + curr_sector
if not os.path.exists(directory):
    os.makedirs(directory)

ticker_list = sec_sym_dic[curr_sector]

api         = API.Tiingo(config)

""" start   = time.time()
price_1 = api.fetch(ticker_list, attr)
end     = time.time()
print('Normal processing time: {time}s.'.format(time=end-start)) """

config['DataAPIFetchMethod'] = 'async'
start   = time.time()
price_2 = api.fetch(ticker_list, attr)
end     = time.time()
print('Asynchronous processing time: {time}s.'.format(time=end-start))

for curr in ticker_list:
    #FileIO.save_csv(price_2[curr], curr  , r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code\STATICS\S&P500Top20')
    FileIO.save_csv(price_2[curr], curr  , directory)
