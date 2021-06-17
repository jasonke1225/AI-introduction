# AI-introduction  
NYCUCS 2021 spring, final project.  
Automated Stock Trading  
Deep reinforce learning with ppo, min variance, DDPG


## Github repo link
https://github.com/teabao/AI-introduction/tree/main/Bao


## Introduction: 

>introduce the problem you want to solve, explain why it is important to solve it; and indicate the method you used to solve it. add a concept figure showing the overall idea behind the method you are presenting


Since cryptocurrency has become the trend of the world, we figure out several ideas to invest in cryptocurrency and try to get as much revenue as we can. The main methods we used are Minimum variance portfolio and the DDPG, PPO algorithms in reinforce learning.

- ### Minimum variance profolio

    <img src="https://i.imgur.com/1IG2Odd.png" width="400">

    Since the portfolios are high dimension data, it's hard to visualize the data. So here is the chart of all portfolios after dimension reduction. Each point on the graph is a indivisual portfolio and the axes are the revennue and variance corresponding to the portfolio.  

    It's worth noting that only the points located on the upper half boundary are efficient, because there will be definitely a better choice than the other points, that is, with same variance but higher revenue.  

    There is a traditional but useful concept that enable us to maximize the revenue with the lowest risk, which is exactly the minimum variance portfolio strategy. The assets in the portfolio may be risky indivisually, but it turns out that the diversity of these assets will be helpful to reduce the risks and get the cross portfolio on the figure, which is the lowest risk portfolio in the efficient points.  

- ### DDPG

- ### PPO


## Related work:

>previous methods that have explored a similar problem


## Bonus: 

>Say why your method is better than previous work; and/or summarize the key main contributions of your work


## Methodology: 
>Details of the proposed technical solution


## Experiments:
>present here experimental results of the method you have implemented with plots, graphs, images and visualizations


## Conclusion: 

>Take home message


## References