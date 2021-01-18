from datetime import datetime
startTime = datetime.now()

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code')

import MAIN.Basics as basics
import MAIN.Reinforcement as RL
import tensorflow as tf
import seaborn as sns
#import matplotlib.pyplot as plt
from UTIL import FileIO
from STRATEGY.Cointegration import EGCointegration
import pickle

# Read config
#config_path  = 'CONFIG\config_train.yml'
config_path  = r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code\CONFIG\config_train.yml'
config_train = FileIO.read_yaml(config_path)

# Read prices
#x = pd.read_csv(r'STATICS\PRICE\JNJ.csv')
#x = pd.read_csv('STATICS\PRICE\JNJ.csv', nrows=5000)# MTR
#x = pd.read_csv(r'STATICS\NewPRICE\FB.csv')
x = pd.read_csv(r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code\STATICS\S&P500\InformationTechnology\V.csv')

#y = pd.read_csv('STATICS\PRICE\PG.csv')
#y = pd.read_csv('STATICS\PRICE\PG.csv', nrows=5000) # MTR
#y = pd.read_csv(r'STATICS\NewPRICE\GOOG.csv') 
y = pd.read_csv(r'C:\DATA\Projects\PortFolioProjects\ThirdProject\Code\STATICS\S&P500\InformationTechnology\MA.csv')

perform_training = False; back_testing = True
x, y = EGCointegration.clean_data(x, y, 'date', 'close')

x.to_excel(r'Results\MTRx.xlsx')  
y.to_excel(r'Results\MTRy.xlsx')
# Separate training and testing sets
train_pct = 0.7
train_len = round(len(x) * train_pct)
idx_train = list(range(0, train_len))
idx_test  = list(range(train_len, len(x)))
EG_Train = EGCointegration(x.iloc[idx_train, :], y.iloc[idx_train, :], 'date', 'close')
EG_Test  = EGCointegration(x.iloc[idx_test,  :], y.iloc[idx_test,  :], 'date', 'close')

# Create action space
n_hist    = list(np.arange(60, 601, 60))
n_forward = list(np.arange(120, 1201, 120))
trade_th  = list(np.arange(1,  5.1, 1))
stop_loss = list(np.arange(1,  2.1, 0.5))
cl        = list(np.arange(0.05,  0.11, 0.05))

actions   = {'n_hist':    n_hist,
            'n_forward': n_forward,
            'trade_th':  trade_th,
            'stop_loss': stop_loss,
            'cl':        cl}
n_action  = int(np.product([len(actions[key]) for key in actions.keys()]))

# Create state space
transaction_cost = [0.001]
states  = {'transaction_cost': transaction_cost}
n_state = len(states)

# Assign state and action spaces to config
config_train['StateSpaceState'] = states
config_train['ActionSpaceAction'] = actions

# Create and build network
one_hot  = {'one_hot': {'func_name':  'one_hot',
                        'input_arg':  'indices',
                        'layer_para': {'indices': None,
                                        'depth': n_state}}}

output_layer = {'final': {'func_name':  'fully_connected',
                        'input_arg':  'inputs',
                        'layer_para': {'inputs': None,
                                        'num_outputs': n_action,
                                        'biases_initializer': None,
                                        'activation_fn': tf.nn.relu,
                                        'weights_initializer': tf.ones_initializer()}}}

layer_dict1 = {'one_hot': {'func_name': 'one_hot',
                            'input_arg': 'indices',
                            'layer_para': {'indices': None,
                                            'depth': n_state}},
                'coint1': {'func_name': 'fully_connected',
                            'input_arg': 'inputs',
                            'layer_para': {'inputs': None,
                                        'num_outputs': 10,
                                        'biases_initializer': None,
                                        'activation_fn': tf.nn.relu,
                                        'weights_initializer': tf.ones_initializer()}}}

layer_dict2 = {'coint2': {'func_name': 'fully_connected',
                                'input_arg': 'inputs',
                                'layer_para': {'inputs': None,
                                                'num_outputs': 10,
                                                'biases_initializer': None,
                                                'activation_fn': tf.nn.relu,
                                                'weights_initializer': tf.ones_initializer()}}}

state_in = tf.placeholder(shape=[1], dtype=tf.int32)

N = basics.Network(state_in)
#N.build_layers(one_hot)
N.build_layers(layer_dict1)
N.add_layer_duplicates(layer_dict2, 3)

N.add_layer_duplicates(output_layer, 1)

if perform_training:
    # Create learning object and perform training
    RL_Train = RL.ContextualBandit(N, config_train, EG_Train)

    sess = tf.Session()
    RL_Train.process(sess, save=True, restore=False, train=True)

    # Extract training results
    action = RL_Train.recorder.record['NETWORK_ACTION']
    reward = RL_Train.recorder.record['ENGINE_REWARD']
    print(np.mean(reward))

    open_file = open(r'Results\net_act_evolution', "wb")
    pickle.dump(RL_Train.net_act_evolution, open_file)
    open_file.close()

    df1 = pd.DataFrame()
    df1['action'] = action
    df1['reward'] = reward
    mean_reward = df1.groupby('action').mean()

    mean_reward_ar = np.array(mean_reward)
    np.savetxt(r'Results\mean_reward_ar.csv', mean_reward_ar, delimiter=',')
    #sns.distplot(mean_reward)
    df1.to_excel(r'Results\df1.xlsx')

    #sns.distplot(mean_reward)

    # Extract opt_action
    #[opt_action] = sess.run([RL_Train.output], feed_dict=RL_Train.feed_dict)
    #opt_action = np.argmax(opt_action)
    #action_dict = RL_Train.action_space.convert(opt_action, 'index_to_dict')
    
    #list_of_poss_state = np.arange(-0.1, 0.1, 0.01)
    column_names = ['n_hist', 'n_forward', 'trade_th', 'stop_loss', 'cl']


    list_of_poss_state = np.arange(-1, 1, 0.1)

    list_of_opt_actions = []
    for possible_state in list_of_poss_state:
        action_dict = RL_Train.action_space.convert(np.argmax(sess.run([RL_Train.output], feed_dict={RL_Train.input_layer:[possible_state]})), 'index_to_dict')        
        if action_dict not in list_of_opt_actions:
            list_of_opt_actions.append(action_dict)
    
    
    print('(list_of_opt_actions)=', (list_of_opt_actions))
    print('len(list_of_opt_actions)=', len(list_of_opt_actions))

    open_file = open(r'Results\list_of_opt_actions', "wb")
    pickle.dump(list_of_opt_actions, open_file)
    open_file.close()
    
    
    sess.close()

if back_testing:
    #action_dict = pd.read_excel('Results\opt_action_df.xlsx', index=False).to_dict()
    #for key in action_dict:
    #    action_dict[key] = action_dict[key][0]
    #
    open_file = open(r'Results\list_of_opt_actions', "rb")
    list_of_opt_actions = pickle.load(open_file)
    open_file.close()

    sess = tf.Session()
    RL_Test = RL.ContextualBandit(N, config_train, EG_Test)
    RL_Test.process(sess, restore=True, train=False)   
    
    #action_dict_dyn = RL_Test.action_space.convert(np.argmax(sess.run([RL_Test.output], feed_dict={RL_Test.input_layer:[0.01]})), 'index_to_dict')
    action_dict = list_of_opt_actions[0]
    # 
    print('action_dict  = ', action_dict)
    # action_dict_dyn={'n_hist':480,'n_forward':1200,'trade_th':2.0,'stop_loss': 1.0, 'cl': 0.1}
    # action_dict_dyn={'n_hist':180,'n_forward':840,'trade_th':2.0,'stop_loss': 1.5, 'cl': 0.05} # the link between _get_network_input and the state broken.
    # 
    #exit()s
    #indices = range(action_dict['n_hist'], len(EG_Test.x) - action_dict['n_forward']) # MTR 
    indices = range(600, len(EG_Test.x) - 1201) # MTR 

    pnl = pd.DataFrame()
    pnl['Time'] = EG_Test.timestamp
    pnl['Trade_Profit'] = 0
    pnl['Cost'] = 0
    pnl['N_Trade'] = 0

    # Lists tht are defined on the entire test dataset, similar to pnl.
    spread_when_there_is_order = []
    buy_sell_order = []

    import warnings
    warnings.filterwarnings('ignore')

    for i in range(0, 600):
        EG_Test.p_values_MTR.append([])
        buy_sell_order.append([])
        spread_when_there_is_order.append([]) 

    for i in indices:
        if i % 100 == 0:
            print('i=', i, ', ', min(indices), '<indices<', max(indices), ', len(EG_Test.x)', len(EG_Test.x)) 
            print(' n_hist, n_forward=', action_dict['n_hist'], action_dict['n_forward'])
        EG_Test.MTR_variables = True

        curr_pri_diff = EG_Test.y[i] - EG_Test.x[i]
        hist_pri_diff = EG_Test.hist_pri_diff
        curr_env_state = (curr_pri_diff/hist_pri_diff) - 1.
        curr_env_state = min(0.1, max(-0.1, curr_env_state))

        action_dict_dyn = RL_Test.action_space.convert(np.argmax(sess.run([RL_Test.output], feed_dict={RL_Test.input_layer:[curr_env_state]})), 'index_to_dict')

        EG_Test.process(index=i, transaction_cost=0.001, **action_dict_dyn)
        ################
        if EG_Test.orders_MTR:
            #if ('Buy' in curr_order) or ('Sell' in curr_order):
            #index_conversion.append(i)        
            buy_sell_order.append(EG_Test.orders_MTR[-1])
            spread_when_there_is_order.append(EG_Test.spread_MTR[-1]) 
        else:
            buy_sell_order.append([])
            spread_when_there_is_order.append([]) 
        ################

        trade_record = EG_Test.record
        if (trade_record is not None) and (len(trade_record) > 0):
            print('value at {}'.format(i))
            trade_record_ori = trade_record
            trade_record = pd.DataFrame(trade_record)
            trade_cost   = trade_record.groupby('trade_time')['trade_cost'].sum() # a pd series.
            # a series is a 1D labeled array. labels are called index. here 'trade_time' will be the index. 
            close_cost   = trade_record.groupby('close_time')['close_cost'].sum()
            profit       = trade_record.groupby('close_time')['profit'].sum()
            # Why for open_pos do we group by 'trade_time' but for close_pos we group by 'close_time'?
            # trade_record.groupby('trade_time') groupby df
            # trade_record.groupby('trade_time')['long_short'] groupby series
            # open_pos/close_pos is a series.
            open_pos     = trade_record.groupby('trade_time')['long_short'].sum()
            close_pos    = trade_record.groupby('close_time')['long_short'].sum() * -1

            # trade_cost.index is essentially trade_record_ori['trade_time']
            # pnl['cost'] is a series. .loc on series is easy.
            # pnl['Time'].isin(trade_cost.index) will be an array containing True/False.
            # to filter one column by multiple values use .loc[...isin()]
            # https://stackoverflow.com/questions/45803676/python-pandas-loc-filter-for-list-of-values
            # pnl['Cost'] is initialized to zero so you are adding trade and close costs.

            pnl['Cost'].loc[pnl['Time'].isin(trade_cost.index)] += trade_cost.values # trade_cost is a pandas series.
            # Why do we have +=?
            pnl['Cost'].loc[pnl['Time'].isin(close_cost.index)] += close_cost.values
            pnl['Trade_Profit'].loc[pnl['Time'].isin(close_cost.index)] += profit.values
            pnl['N_Trade'].loc[pnl['Time'].isin(trade_cost.index)] += open_pos.values
            pnl['N_Trade'].loc[pnl['Time'].isin(close_cost.index)] += close_pos.values

            tmp = 1

    for i in range(len(EG_Test.timestamp) - action_dict['n_forward'], len(EG_Test.timestamp)):
        EG_Test.p_values_MTR.append([])
        buy_sell_order.append([])
        spread_when_there_is_order.append([]) 

    warnings.filterwarnings(action='once')
    print('Finished testing')

    pnl['PnL'] = (pnl['Trade_Profit'] - pnl['Cost']).cumsum()
    pnl.to_excel('Results\pnl.xlsx')

    #p_values_arr = np.array(EG_Test.p_values_MTR)
    #np.savetxt(r'Results\p_values_arr.csv', p_values_arr, delimiter=',')


    open_file = open(r'Results\p_values', "wb")
    pickle.dump(EG_Test.p_values_MTR, open_file)
    open_file.close()

    open_file = open(r'Results\buy_sell_order', "wb")
    pickle.dump(buy_sell_order, open_file)
    open_file.close()

    open_file = open(r'Results\spread_when_there_is_order', "wb")
    pickle.dump(spread_when_there_is_order, open_file)
    open_file.close()

    print(pnl['PnL'].iloc[[-1]])

print('datetime.now() - startTime = ', datetime.now() - startTime) 
