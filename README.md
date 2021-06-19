# AI-introduction  
NYCUCS 2021 spring, final project.  
Automated Stock Trading  
Deep reinforce learning with ppo, min variance, DDPG


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

- ### DDPG

- ### PPO


## Related work:

>previous methods that have explored a similar problem


## Bonus: 

>Say why your method is better than previous work; and/or summarize the key main contributions of your work


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

- ### DDPG

- ### PPO


## Experiments:
>present here experimental results of the method you have implemented with plots, graphs, images and visualizations

- ### Minimum variance portfolio

    As we know that there's not a better way to visualize the data in four-dimensional space, we will plot the portfolios in the form of revenues and variance. 
    
    <div style="text-align:center">
        <img src="https://i.imgur.com/1IG2Odd.png" width="300" height="300">
        <img src="https://i.imgur.com/YM9WWZQ.png" width="300" height="300">
    </div>

    Since it's hard to find out all the possible portfolio, we randomly sample 10000 portfolios and plot their corresponding revenue and variance on the right graph. The filled region is the rough range decided by those sample points. However, as shown in the graph, the region is not like the left graph that we expect to have. I think the main reason is that the domain of our investment only contains 3 types of cryptocurrency, and these 3 cryptocurrencies are highly dependent to each other. 

    <div style="text-align:center">
        <img src="https://i.imgur.com/lQvuTTq.png" width="300" height="300">
    </div>

    By concept of gradient descent, we can find the portfolio with minimum variance as shown in the graph, the red lines stand for the moving path with different initial weights. We finally obtain the upper left point which is the portfolio with weight $w = (w_{Bitcoin},w_{Ethereum},w_{Monero}) \approx (0.8547,0.1453,0)$ and its corresponding variance $V \approx 0.002012559294621305$ 

    Notice that the main idea to take minimum variance portfolio as strategy is trying to reduce the risk by holding individual assets. However, due to the lack of diversity and independence in this case, the strategy here decides to hold only Bitcoins because it has the smaller variance but higher expected revenue.
    
    <div style="text-align:center">
        <img src="https://i.imgur.com/pVp2Du6.png" width="300">
    </div>

    After trading for one years, the money changes from 1,000,000 dollars to 7,138,033 dollars. The return on investment using minimum variance portfolio strategy is over 600%. 


- ### DDPG

- ### PPO


## Conclusion: 

>Take home message

The minimum variance portfolio doesn't work very well in the market of cryptocurrency because its conservatism and lack of flexibility.

On the other hand, the DDPG and PPO .........

## References