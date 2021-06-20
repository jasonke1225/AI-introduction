# AI-introduction  
NYCUCS 2021 spring, final project.  
Automated Stock Trading  
Deep reinforcement learning with ppo, min variance, DDPG


## Github repo link
https://github.com/teabao/AI-introduction/tree/main/Bao


## Introduction: 
 
>introduce the problem you want to solve, explain why it is important to solve it; and indicate the method you used to solve it. add a concept figure showing the overall idea behind the method you are presenting


Since cryptocurrency has become the trend of the world, we figure out several ideas to invest in cryptocurrency and try to get as much revenue as we can. The main methods we used are Minimum variance portfolio and the DDPG, PPO algorithms in reinforce learning.

In this project, we are engaging in the investment of 3 cryptocurrency. They are respectively Bitcoin, Ethereum and Monero.



- ### Minimum variance portfolio

    <img src="https://i.imgur.com/1IG2Odd.png" width="400">

    Since the portfolios are high dimension data, it's hard to visualize the data. So here is the chart of all portfolios after dimension reduction. Each point on the graph is a individual portfolio and the axes are the revenue and variance corresponding to the portfolio.  

    It's worth noting that only the points located on the upper half boundary are efficient, because there will be definitely a better choice than the other points, that is, with same variance but higher revenue.  

    There is a traditional but useful concept that enable us to maximize the revenue with the lowest risk, which is exactly the minimum variance portfolio strategy. The assets in the portfolio may be risky individually, but it turns out that the diversity of these assets will be helpful to reduce the risks and get the cross portfolio on the figure, which is the lowest risk portfolio in the efficient points.  

- ### PPO

    <img src="https://i.imgur.com/FUCYg0Z.jpg" width="600">


    Deep reinforcement learning(DRL) is applied in lots of fields and has a nice performace today, therefore, we think it may be a good try to apply DRL to cryptos. The reason choosing PPO as one of our algorithms is due to its charachteristic of actor-critc approach fitting in a continuos action space. Nowadays, Actor-critic approaches are widely used in financial technogy. In addition, because of PPO's unique updating method minimizing the clipped and normal objective, the stableness of PPO also makes it a good candidate to apply.


- ### DDPG

    ![](https://i.imgur.com/fxLmcZ8.png)

    Deep Deterministic Policy Gradient(DDPG) is one of the off-policy reinforment learning algorithm which can deal with the high-dimension action space. With the characheristic of off-policy learning as DQN algorithm, it use the replay buffer to minimize the correlation of the training data and estimate the Q value of the policy. The actor-critic structure, in constrast, provide it the ability to handle the high-dimension and continuous action space as well as to update the police.  
    
    The reason we choose DDGP as the research method is because of its performance in previous research where it has a intermediate performance of all methods. We expect to observe a similar result corresponding to all the other method we apply in this environment.  
    


## Related work:

>previous methods that have explored a similar problem
- We refer to the paper, "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy". It used mutiple agorithms in stock trading. The main purpose is to compare the policy generated from the agorithms it choose in stock.
- The result of the research is that PPO reachs the best performance; MSE strategy reachs the worst; DDPG has the middle performance between the two.
- Therefore, we choose the three algorithm as our research method on cryptocurrency and use the similiar environment to train our model.


## Bonus: 

>Say why your method is better than previous work; and/or summarize the key main contributions of your work


- The environment state in the paper seems to deal without preprocessing, which may be a huge burden to the training of neural network. Moreover, the change of the cryptocurrency is much more considerable, which might lead to the ploicy explosion in our neural network. We, as the result normalized and reweighted the states to facilitate the training. Due to the preprocessing method we do, it really improves our training process, especially in PPO. 


## Methodology: 
>Details of the proposed technical solution

- ### Minimum variance portfolio

    To estimate the future revenue of a specific cryptocurrency, we have to observe price changes in the past. Since we have the daily price of each cryptocurrency, we can derive the percentage change in logarithm (In order to prevent the result from getting small)
    
    $$ CoV(X, Y) = E[ (X-E[X]) (Y-E[Y]) ] $$
    
    Covariance is defined as the expected value of the product of the two individual deviation. So, we can apply the formula to obtain the covariance between each pair of cryptocurrencies, which stand for the joint variability of two cryptocurrency.

    $$ V(portfolio) = \sum_{X,Y} Weight(X) * Weight(Y) * CoV(X,Y) \\
                    = \sum_i W_i*(C_{0i}W_0+C_{1i}W_1+C_{2i}W_2)$$

    Since the final goal is to find a portfolio that has the minimum variance, we have to define what is the variance of a portfolio. Here we just take the weighted sum of all covariances as the variance of portfolio intuitively. 

    <img src="https://i.imgur.com/5grGJ85.png" width="600">

    To minimize the variance, we decide to adopt the concept of gradient descent, which is mentioned in the class. Therefore, it means that we should find the gradient of variance function $V(w_{Bitcoin},w_{Ethereum},w_{Monero})$, where $w_s$ is the corresponding weight of a cryptocurrency in the portfolio. In the above computation graph, we can derive the gradient of each $w_s$: 
    
    $$\frac{{\rm d}V}{{\rm d}w_s} = 2𝐶_{0,s} w_0 + 2𝐶_{1,s} w_1 + 2𝐶_{2,s} w_2$$

    Then, we implement the concept of gradient descent to look for the portfolio that reach the minimum variance step by step.

    1. Start with weight $w = (w_{Bitcoin},w_{Ethereum},w_{Monero}) = (\frac{1}{3},\frac{1}{3},\frac{1}{3})$ with initial step size $\eta=0.1$
    2. Compute the gradient $\nabla_wV(w)$ with the weight.
    3. Update the weight by subtracting $\eta * \nabla_wV(w)$.
    4. Compute the corresponding variance of the weight.
    5. If the variance does not converge, go to step 2.
    6. Obtain the portfolio with minimum variance.  

    However, the final portfolio we chose might be a local minimum rather than the global minimum. So we will randomly pick some weights to be the initial weight and follow the same procedure trying to prevent such situation.

    
- ### PPO
    
    PPO, Proximal Policy Optimization, evolves from Trust Region Policy Optimization. In this project, instead of using the adatptive KL penalty version of PPO, we used the clipped surragate objective version of PPO.  
    
    First,in PPO, we collect trajectories under a policy $\pi_k = \pi(\theta_{k})$ and estimate its advantage by using an advantage estimate function. Here, we use:
    $$A^{\pi_k} = G_{t} - V^{\pi_k}$$
    $$A: advantage, G:accumulated\:reward,V:state\:value$$
    Next, we take the minimum from the two surrogate functions and get its expected value. After that, we get the arguments of the maxima to update our policy parameters $\theta$. We repeat this step for $K$ times. 
    
    The PPO algorithm is as follow:


    <img src="https://i.imgur.com/m3hYPKf.png" width="800">
    
    (Picture Credit: Katerina Fragkiadaki))

    
- ### DDPG

    Taking the advantage Actor-Critic, DDPG apply the Actor Network to choose the action, the Critic Network to evaluate the Q value of which. For example, when a state comes to the Actor Network in a six-dimension action space environment, it will return six action respectively to each diamension. Then, sending the state and the action generated from the Actor Network to the Critic Network, it will generate a Q value.  
    
    Due to its off-policy character, after getting the action from Actor Network, we do not directly interact with the environment with the action. It need the addition of the Randomness on action to improve the exploration rate.
    
    Further, since the Q value it returns, DDPG do the similiar method to minimize loss on Critic Network in temporal difference as DQN with the replay buffer:
    $$L = \mathbb{E}[(r_i + \gamma Q^{'}(s_{i+1},\mu^{'}(s_{i+1})) - Q(s_i,a_i)) ^{2}] $$
    $$Q: critic\:network,Q^{'}: target\:critic\:network, \:\mu^{'}:target\:actor\:network,\:$$
    And update the Actor Network by the method in deterministic policy gradient(DPG). Therefore, DDPG maintain a current network and the target network for both network mentioned above like what DQN does but update the target network by degrees for the sake of stability.
    
    At last, the most important and difficult point is to initialize the parameter in the neural network. With the appropriate initialization, the algorithm could efficiently converge to the optimal policy, or the algorithm might never converge. It is much more sensitive than other two algorithm. 
    
    The DDPG algorithm is as follow:
    
    ![](https://i.imgur.com/tLvfNhh.png)




## Experiments:
>present here experimental results of the method you have implemented with plots, graphs, images and visualizations

1.  Training Data : 2015/08/08 ~ 2019/05/14
1.  Testing Data : 2019/05/14 ~ 2021/05/14
1.  cryptocurrency value : 
    | Testing Date | 2019/05/14 | 2021/05/14 | ratio |
    | ------------ | ---------- | ---------- |------ |
    |   Bitcoin    | 248794.81  | 1393263.01 | 560%  |
    |   Bitcoin    | 6757.9     | 113936.23  | 1686% |
    |   Bitcoin    | 2585.37    | 11523.08   | 446%  |


- ### Minimum variance portfolio
    <div style="text-align:center">
        <img src="https://i.imgur.com/1IG2Odd.png" width="300" height="300">
        <img src="https://i.imgur.com/YM9WWZQ.png" width="300" height="300">
    </div>

    Since it's hard to find out all the possible portfolio, we randomly sample 10000 portfolios and plot their corresponding revenue and variance on the right graph. The filled region is the rough range decided by those sample points. However, as shown in the graph, the region is not like the left graph that we expect to have. I think the main reason is that the domain of our investment only contains 3 types of cryptocurrency, and these 3 cryptocurrencies are highly dependent to each other. 

    <div style="text-align:center">
        <img src="https://i.imgur.com/lQvuTTq.png" width="300" height="300">
    </div>

    By concept of gradient descent, we can find the portfolio with minimum variance as shown in the graph, the red lines stand for the moving path with different initial weights. We finally obtain the upper left point which is the portfolio with weight $w =(w_{Bitcoin},w_{Ethereum},w_{Monero}) \approx (0.8547,0.1453,0)$ and its corresponding variance $V \approx 0.002012559294621305$

    Notice that the main idea to take minimum variance portfolio as strategy is trying to reduce the risk by holding individual assets. However, due to the lack of diversity and independence in this case, the strategy here decides to hold Bitcoins the most because it has the smaller variance but higher expected revenue.
    
    <div style="text-align:center">
        <img src="https://i.imgur.com/YRExY1s.png" width="300">
    </div>

    After trading for two years, the money changes from 1,000,000 dollars to 7,138,033 dollars. The return on investment using minimum variance portfolio strategy is about 610%. 


- ### PPO
    Because of the limitation of our training data, the result of PPO seems not as good as we expected. ~~After training our model by using the data from 2015 to 2019, we test the model between 2019 and 2021.~~ The following fugure is the result of the total asset:

    <div style="text-align:center">
        <img src="https://i.imgur.com/UtVgkMX.png" width="350">
    </div>
    After trading for two years, the toatal asset changes from 1 million to 1.4 million. The return ratio of PPO is about 40%.
    

- ### DDPG

     DDPG shows a good performance. In the end of the test, its total asset ascends to 7.3 million. The return on investment is bout 630%. 
    <div style="text-align:center">
        <img src="https://i.imgur.com/qEg5TY7.png" width="300">
    </div>
    
    To observe the action in the trajectory in detail, DDPG attachs importance to Bitcoin, the main operating is on the one. Here, we guess it is according to the variance of Bitcoin in traing data is high. It, hence, pays attention to Bitcoin.
    
    
    
## Conclusion: 

>Take home message

The minimum variance portfolio doesn't work very well in the market of cryptocurrency because its conservatism and lack of flexibility.

On the other hand, the DDPG and PPO .........

## References
Hongyang Yang1, Xiao-Yang Liu2, Shan Zhong2, and Anwar Walid3. Deep Reinforcement Learning for Automated
Stock Trading: An Ensemble Strategy. 2020.


Timothy P. Lillicrap∗, Jonathan J. Hunt∗, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra, CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING. In *ICLR*, 2016.

