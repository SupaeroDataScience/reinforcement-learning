### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

config = {'observation_space': cartpole.observation_space.shape[0],
          'nb_actions': cartpole.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 20}

agent = DQN_agent(config, DQN)
scores = agent.train(cartpole, 200)
plt.plot(scores)

x = cartpole.reset()
cartpole.render()
for i in range(1000):
    a = greedy_action(DQN, x)
    y, _, d, _ = cartpole.step(a)
    cartpole.render()
    x=y
    if d:
        print(i)
        break

cartpole.close()