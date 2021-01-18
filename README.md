# Pair-Trading-Reinforcement-Learning
My contributions to this code are as follows.

1)	fixing a bug in the original repo
2)	allowing the user to generate reproducible results 
3)	extending the code from stateless to contextual bandit
4)	separating training and back-testing sections of the code.

For the details of the critical bug and my solution, 
see the issue opened in the original repo. 
https://github.com/wai-i/Pair-Trading-Reinforcement-Learning/issues/8. 

In the original repo, random number generators were not 
properly seeded, so the results were not reproducible. 
This commit fixed that problem.

Extending from stateless to contextual bandit allows 
the agent to adjust the trading parameters according 
to the environment's current state. 
In the committed version, the environment can have 
only two states: positive or negative (depending on 
whether the spread is above or below the historical value). 
In the tests I performed before making this commit, 
the optimal parameters learned by the agent with binary 
environment representation were virtually the same as stateless. 
In future versions, the environment should be 
allowed to have more states. 

I separated training and back-testing parts because it 
allows one to isolate potential problems better. 
At the end of the training, the TensorFlow graph is saved 
and then restored in back-testing.

Also, price data for the top 50 indices in each of the 
top three S&P500 sectors are downloaded (from Tiingo) 
into the working directory. 


<p align="center">
  <img width="600" src="Structure.PNG">
</p>
<p align="justify">

A TensoFlow implemention in Reinforcement Learning and Pairs Trading. The current status of the project covers implementation of RL in cointegration pair trading based on 1-minute stock market data. For the Reinforcement Learning here we use the N-armed bandit approach. The code is expandable so you can plug any strategies, data API or machine learning algorithms into the tool if you follow the style.

## Guidance

* [Medium](https://medium.com/@wai_i/a-gentle-implementation-of-reinforcement-learning-in-pairs-trading-6cdf8533bced) - The corresponding article for this project.

## Data Source
* [Tiingo](https://www.tiingo.com/) - A financial research platform that provides data including news, fundamentals and prices. We can extract the intraday stock market data through its REST IEX API that retrieves TOPS data (top of book, last sale data and top bid and ask quotes) from the IEX Exchange.

## Examples
See the folder EXAMPLE for more detail. Please initiate your own virtual environment before running the code.

## Disclaimer
The article and the relevant codes and content are purely informative and none of the information provided constitutes any recommendation regarding any security, transaction or investment strategy for any specific person. The implementation described in the article could be risky and the market condition could be volatile and differ from the period covered above. All trading strategies and tools are implemented at the users’ own risk.
