## Reinforcement Learning Algorithms
Implemented Q-Learning, SARSA, and Cross Entropy Method using numpy and torch and compared their performance on frozenlake-deterministic, frozenlake-stochastic, and cliffwalking.

## Dependencies
* OpenAI gym
* matplotlib
* numpy
* collections
* torch
* itertools
* plotting

## Deterministic Frozenlake Results
![alt text](https://github.com/Phrungck/reinforcement-learning-models/blob/main/images/frozen-det.PNG)

## Stochastic Frozenlake Results
![alt text](https://github.com/Phrungck/reinforcement-learning-models/blob/main/images/frozen-sto.PNG)

## Cliffwalking Results
![alt text](https://github.com/Phrungck/reinforcement-learning-models/blob/main/images/cliffwalk.PNG)

## Changing Parameters
![alt text](https://github.com/Phrungck/reinforcement-learning-models/blob/main/images/comp-frozen-sto.PNG)

All results showed that SARSA and Q-Learning bested Cross-entropy method for the CliffWalking environment. Changes in the hyperparameters showed significant changes. Notably, by increasing the alpha parameter Q-Learning and SARSA exceeded results of the baseline. 

Increase in alpha while reducing Gamma resulted to almost similar values for all variants of Q-Learning and SARSA. However, Cross-entropy became more erratic in the process.
