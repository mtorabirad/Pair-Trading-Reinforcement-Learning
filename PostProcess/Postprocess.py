# Items: 
import pandas as pd
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from pdf2image import convert_from_path, convert_from_bytes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from numpy import genfromtxt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
FontSize = 12
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : FontSize}
plt.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"

spread_in_one_inx = False; prices = False; mean_reward = False; plotpnl = True; plot_act_evolution = False

x_lab = 'V'; y_lab = 'MA'
x = pd.read_excel(r'Results\MTRx.xlsx')
y = pd.read_excel(r'Results\MTRy.xlsx')
train_len = round(len(x) * 0.7)
idx_train = list(range(0, train_len))
idx_test  = list(range(train_len, len(x)))
x_train = x.iloc[idx_train, :]
y_train = y.iloc[idx_train, :]
x_test = x.iloc[idx_test, :]
y_test = y.iloc[idx_test, :]


print('len(x_train)=', len(x_train))
print('len(x_test)=', len(x_test))

""" 
opt_action_df = pd.read_excel(r'Results\opt_action_df.xlsx')

print('opt_action_df=', opt_action_df)

trade_th = opt_action_df.loc[[0],'trade_th'][0]
stop_loss  = trade_th + opt_action_df.loc[[0],'stop_loss'][0]
n_forward = opt_action_df.loc[[0],'n_forward'][0]
n_hist = opt_action_df.loc[[0],'n_hist'][0] 
"""
import pickle
open_file = open(r'Results\list_of_opt_actions', "rb")
list_of_opt_actions = pickle.load(open_file)
open_file.close()
print('list_of_opt_actions[1]=', list_of_opt_actions[1])

pnl = pd.read_excel(r'Results\pnl.xlsx')

import pickle
open_file = open(r'Results\p_values', "rb")
p_values = pickle.load(open_file)
open_file.close()

open_file = open(r'Results\buy_sell_order', "rb")
buy_sell_order = pickle.load(open_file)
open_file.close()

open_file = open(r'Results\spread_when_there_is_order', "rb")
spread = pickle.load(open_file)
open_file.close()

print('max(N_Trade)=', pnl['N_Trade'][pnl['N_Trade'].idxmax()])
print('min(N_Trade)=', pnl['N_Trade'][pnl['N_Trade'].idxmin()])

### Find the window with the highest number of sells
max_num_sells = 0
index_with_max_sells = 0
def check_condition(x, condition): return x == condition
#for i in range(n_hist, len(x_test) - n_forward):
for i in range(600, len(x_test) - 1201):
    Sell_index = np.array([idx + 1 + i for idx, element in enumerate(buy_sell_order[i]) if check_condition(element, 'Sell')])
    if len(Sell_index) > max_num_sells:
        max_num_sells =  len(Sell_index)
        index_with_max_sells = i
print('max_num_sells=', max_num_sells)
print('index_with_max_sells=', index_with_max_sells)

###
if spread_in_one_inx:

    ###############################################################
    max_num_buys = 0
    index_with_max_buys = 0
    #for i in range(n_hist, len(x_test) - n_forward):
    for i in range(600, len(x_test) - 1201):
        Buy_index = np.array([idx + 1 + i for idx, element in enumerate(buy_sell_order[i]) if check_condition(element, 'Buy')])
        if len(Buy_index) > max_num_buys:
            max_num_buys =  len(Buy_index)
            index_with_max_buys = i
    print('max_num_buys=', max_num_buys)
    print('index_with_max_buys=', index_with_max_buys)

    max_num_stops = 0
    index_with_max_stops = 0
    #for i in range(n_hist, len(x_test) - n_forward):
    for i in range(600, len(x_test) - 1201):
        Stop_index = np.array([idx + 1 + i for idx, element in enumerate(buy_sell_order[i]) if check_condition(element, 'Stop')])
        if len(Stop_index) > max_num_stops:
            max_num_stops =  len(Stop_index)
            index_with_max_stops = i
    print('max_num_stops=', max_num_stops)
    print('index_with_max_stops=', index_with_max_stops) 

    another_index = 0
    #for i in range(n_hist, len(x_test) - n_forward):
    for i in range(600, len(x_test) - 1201):
        Stop_index = np.array([idx + 1 + i for idx, element in enumerate(buy_sell_order[i]) if check_condition(element, 'Stop')])
        Buy_index = np.array([idx + 1 + i for idx, element in enumerate(buy_sell_order[i]) if check_condition(element, 'Buy')])
        Sell_index = np.array([idx + 1 + i for idx, element in enumerate(buy_sell_order[i]) if check_condition(element, 'Sell')])
        if (len(Stop_index) > 2) and (len(Buy_index) > 2) and (len(Sell_index) > 2):
            another_index = i
    print('another_index=', another_index)
    ###############################################################
    nrows_ = 3; ncols_ = 1
    fig, axes = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(10,7.5))

    which_indexs = [index_with_max_buys, index_with_max_stops, index_with_max_sells]
    i = 0
    for which_index in which_indexs:
        x_forward = range(which_index, which_index + n_forward)
        axes[i].plot(x_forward, spread[which_index], alpha=0.5)
        axes[i].hlines(trade_th, min(x_forward), max(x_forward))
        axes[i].hlines(-trade_th, min(x_forward), max(x_forward))
        axes[i].hlines(stop_loss, min(x_forward), max(x_forward))
        axes[i].hlines(-stop_loss, min(x_forward), max(x_forward))

        color_dic = {'Sell':'r', 'Buy': 'b', 'Stop':'k'}
        #def check_condition(x, condition): return x == condition

        Sell_index = np.array([idx + 1 + which_index for idx, element in enumerate(buy_sell_order[which_index]) if check_condition(element, 'Sell')])
        try:
            axes[i].scatter(Sell_index, spread[which_index][Sell_index - which_index], marker='o', c=color_dic['Sell'])
        except:
            pass

        Buy_index = np.array([idx + 1 + which_index for idx, element in enumerate(buy_sell_order[which_index]) if check_condition(element, 'Buy')])
        try:
            axes[i].scatter(Buy_index, spread[which_index][Buy_index - which_index], marker='o', c=color_dic['Buy'])
        except:
            pass

        Stop_index = np.array([idx + 1 + which_index for idx, element in enumerate(buy_sell_order[which_index]) if check_condition(element, 'Stop')])
        axes[i].scatter(Stop_index, spread[which_index][Stop_index - which_index], marker='o', c=color_dic['Stop'])
        i +=1
        #axes[i].title('index='+str(which_index))
    
    plt.tight_layout()
    plt.show()
    fig.savefig('spread_in_one_inx')

if prices:
    nrows_ = 2; ncols_ = 2
    fig, axes = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(8,8))
    axes[0,0].scatter([i for i in range(len(x_train['close']))], x_train['close'], s=2, c='k', label=x_lab)
    axes[0,0].scatter([i for i in range(len(y_train['close']))], y_train['close'], s=2, c='b', label=y_lab)
    axes[0,0].scatter([i for i in range(len(x_train['close']), len(x['close']))], x_test['close'], s=2, c='k')
    axes[0,0].scatter([i for i in range(len(y_train['close']), len(y['close']))], y_test['close'], s=2, c='b')
    axes[0,0].vlines(x=max(idx_train), ymin=0, ymax=max(max(x_train['close']), max(y_train['close'])), linestyles='dotted')
    axes[0,0].set_ylabel('Raw prices')
    axes[0,0].legend()
    #axes[0,0].text(0.5, 0.5, "{:0.4f}".format(p_val), transform=axes[0,0].transAxes)

    reg = LinearRegression().fit(np.array(x['close']).reshape(-1, 1), np.array(y['close']).reshape(-1, 1))
    beta = reg.coef_[0]
    resid = np.array(y['close']) - np.array(x['close']) * beta

    scaled_price_difference = resid
    spread = (scaled_price_difference - scaled_price_difference.mean()) / scaled_price_difference.std()


    axes[0,1].scatter([i for i in range(len(scaled_price_difference))], scaled_price_difference, s=2, c='k')
    axes[0,1].set_ylabel('Scaled price difference')
    
    spread_tra = spread[0:len(idx_train)]

    axes[1,0].scatter([i for i in range(len(spread_tra))], spread_tra, s=2, c='k')
    axes[1,0].hlines(trade_th, xmin=0, xmax=len(spread))
    axes[1,0].hlines(-trade_th, xmin=0, xmax=len(spread))
    axes[1,0].hlines(stop_loss, xmin=0, xmax=len(spread))
    axes[1,0].hlines(-stop_loss, xmin=0, xmax=len(spread))
    #axes[1,0].text(0.1, 0.1, 'beta='+str(beta), fontsize=12, transform=axes[1,0].transAxes)
    axes[1,0].set_ylabel('Spread (Z-score) - Training')

    spread_backtest = spread[len(idx_train):len(spread)]

    axes[1,1].scatter([i for i in range(len(idx_train),len(spread))], spread[idx_test], s=2, c='k')
    axes[1,1].set_ylabel('Spread (Z-score) - Backtesting')
    plt.tight_layout()
    fig.savefig('prices')

if mean_reward:
    mean_reward = genfromtxt('mean_reward_ar.csv', delimiter=',')
    sns.distplot(mean_reward) 
    fig.savefig('mean_reward')

if plotpnl:
    
    nrows_ = 2; ncols_ = 2
    fig, axes = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(8,8))

    axes[0,0].scatter([i for i in range(len(pnl['N_Trade']))], pnl['N_Trade'], s=2, c='k')
    axes[0,0].set_ylabel('Total Number of Trades at Each Index')

    axes[0,1].scatter([i for i in range(len(pnl['Trade_Profit']))], pnl['Trade_Profit'], s=2, c='k')
    axes[0,1].set_ylabel('Profit Associated with the Total Trades')

    axes[1,0].scatter([i for i in range(len(pnl['Cost']))], pnl['Cost'], s=2, c='k')
    axes[1,0].set_ylabel('Cost Associated with the Total Trades')  

    axes[1,1].scatter([i for i in range(len(pnl['PnL']))], pnl['PnL'], s=2, c='k')
    axes[1,1].set_ylabel('Profit and Loss (PnL)'); 

    for i in range(0,nrows_):
        for j in range(ncols_):
            axes[i,j].set_xlabel('Time Index'); 
            axes[i,j].xaxis.set_major_locator(MultipleLocator(5000)); axes[i,j].xaxis.set_minor_locator(MultipleLocator(2500))
            #axes[i,j].yaxis.set_major_locator(MultipleLocator(200)); axes[i,j].yaxis.set_minor_locator(MultipleLocator(100))
            axes[i,j].tick_params(direction='in', length=6, width=1, colors='k', grid_color='k', grid_alpha=0.5, which='major', labelsize=FontSize)
            axes[i,j].tick_params(direction='in', length=3, width=0.5, colors='k', grid_color='k', grid_alpha=0.5, which='minor', labelsize=FontSize)
    plt.tight_layout()
    #plt.suptitle('Pairs:' + x_lab + ', ' + y_lab)
    fig.savefig(r'Results\plotpnl')

if plot_act_evolution:
    open_file = open(r'Results\net_act_evolution', "rb")
    net_act_evolution = pickle.load(open_file)
    open_file.close()
    for curr_epi in range(len(net_act_evolution)):
        parameter = 'n_hist'
        #parameter = 'n_forward'
        list_for_this_para = [curr_dic[parameter] for curr_dic in net_act_evolution[curr_epi]]
        #plt.scatter([i for i in range(len(net_act_evolution[curr_epi]))], net_act_evolution[curr_epi], label='episode='+str(curr_epi), s=2)
        plt.scatter([i for i in range(len(list_for_this_para))], list_for_this_para, label='episode='+str(curr_epi), s=2)
    plt.legend()
    plt.show()

#plt.show()