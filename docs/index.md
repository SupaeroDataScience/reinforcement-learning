# Reinforcement Learning
Reinforcement Learning module of the Machine Learning class at ISAE-Supaero.

[Github repository](https://github.com/SupaeroDataScience/reinforcement-learning/)

## Syllabus

This class covers an introduction to Reinforcement Learning (RL) in 15 hours, over 5 sessions. It aims to provide both a solid theoretical foundation and a quick learning curve towards current Deep RL algorithms. It starts with the fundamental notions underlying RL: Markov Decision Processes, model-based resolution approaches including Dynamic Programming, sample-based resolution of the Bellman equation. This leads to the identification of the three bottomline challenges in RL: function approximation, the exploration/exploitation trade-off and the search for optimality. This provides perspective to the following classes that introduce methods designed to tackle these challenges, including Deep RL methods. By the end of the class, students should be able to understand the literature on RL, implement key algorithms, and anticipate the difficulties of applying RL to various problems.



## Class material

The class is split into a series of notebooks that serve as lecture material, textbook and exercice book.

Open the notebooks in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SupaeroDataScience/reinforcement-learning/HEAD)


## Class schedule

Schedule | | | |
--- | --- | --- | ---
11/01 PM| Introduction and MDPs | RL intuitions, Robotics, Markov Decision Processes |
18/01 PM| Bellman Equations and Value Functions |  Bellman Equations, Characterizing and Evaluating Policies |
25/01 PM| Deep Q-Networks | Value Function Approximation, Deep Q-Networks |
01/02 PM| Deep Q-Networks (lab) |  Hands-on Implementation of DQN |
08/02 PM| Actor-Critic methods | Policy Gradients and Actor-Critic Algorithms |


### Introduction to Reinforcement Learning
--8<--
docs/RL0-intro.md
--8<--

### Markov Decision Processes
--8<--
docs/RL1-MDP.md
--8<--

### Bellman equations, characterizing optimal policies
--8<--
docs/RL2-DP.md
--8<--

### Evaluating policies with samples
--8<--
docs/RL3-eval.md
--8<--

### Solving the optimality equation with samples
--8<--
docs/RL4-control.md
--8<--

### Deep Q-Networks
--8<--
docs/RL5-DQN.md
--8<--


## Additional resources

Great books available online:  
[Reinforcement Learning, an introduction](http://incompleteideas.net/book/the-book.html)  
[Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/RLBook.html)  
[An introduction to Deep Reinforcement Learning](https://arxiv.org/abs/1811.12560)

[FAQ on installing Gym for Mac users](https://github.com/SupaeroDataScience/reinforcement-learning/blob/master/notebooks/extras/Install_RLclass_MacOS.md)

The [AlphaGo movie](https://www.youtube.com/watch?v=WXuK6gekU1Y)

