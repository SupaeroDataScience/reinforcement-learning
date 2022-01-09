### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

# Let's restart from Qpi0
Qsarsa = Qpi0
max_steps = 5000000

def epsilon_greedy(Q, s, epsilon):
    a = np.argmax(Q[s,:])
    if(np.random.rand()<=epsilon): # random action
        aa = np.random.randint(env.action_space.n-1)
        if aa==a:
            a=env.action_space.n-1
        else:
            a=aa
    return a

# SARSA
count = np.zeros((env.observation_space.n,env.action_space.n)) # to track update frequencies
epsilon = 1
x = env.reset()
a = epsilon_greedy(Qsarsa,x,epsilon)
for t in range(max_steps):
    if((t+1)%1000000==0):
        epsilon = epsilon/2
    y,r,d,_ = env.step(a)
    aa = epsilon_greedy(Qsarsa,y,epsilon)
    Qsarsa[x][a] = Qsarsa[x][a] + alpha * (r+gamma*Qsarsa[y][aa]-Qsarsa[x][a])
    count[x][a] += 1
    if d==True:
        x = env.reset()
        a = epsilon_greedy(Qsarsa,x,epsilon)
    else:
        x=y
        a = aa

# SARSA's final value function and policy
print("Max error:", np.max(np.abs(Qsarsa-Qstar)))

print("Final epsilon:", epsilon)
pi_sarsa = greedyQpolicy(Qsarsa)
print("Greedy SARSA policy:")
print_policy(pi_sarsa)
print("Difference between pi_sarsa and pi_star (recall that there are several optimal policies):")
print(pi_sarsa-pi_star)
Qpi_sarsa, residuals = policy_Qeval_iter(pi_sarsa,1e-4,10000)
print("Max difference in value between pi_sarsa and pi_star:", np.max(np.abs(Qpi_sarsa-Qstar)))
print("Min difference in value between pi_sarsa and pi_star:", np.min(np.abs(Qpi_sarsa-Qstar)))