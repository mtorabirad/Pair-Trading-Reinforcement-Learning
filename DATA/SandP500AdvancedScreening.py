# https://tcoil.info/how-to-get-list-of-companies-in-sp-500-with-python/
# This file downloads the S&P500 symbols and arranges them into a dictionary {sector:[symbol list]}

import pandas as pd

# There are 2 tables on the Wikipedia page
# we want the first table

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]

df = first_table
symbols = df['Symbol'].values.tolist()

sectors = df['GICS Sector'].values.tolist()
sectors = set(sectors)

sec_sym_dic = {}

Health_syms = df[df['GICS Sector']=='Health Care']['Symbol'].values.tolist()
for curr_sec in sectors:
    curr_syms = df[df['GICS Sector']==curr_sec]['Symbol'].values.tolist()
    sec_sym_dic[curr_sec] = curr_syms

print(sec_sym_dic['Information Technology'])

import pickle
def save_dic(obj, name ):
    with open(r'DATA/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_dic(sec_sym_dic, 'sec_sym_dic')