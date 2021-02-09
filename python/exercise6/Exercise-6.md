# Exercise-6

After trying different grid-sizes and timesteps, our conclusion is that both vary grid-size and timesteps have nearly the same effects (for our choice of grid-sizes and timesteps).
We tried among other sizes and timesteps (60 timesteps and a 10x10 grid, 30 timesteps and a 28x28 grid) at which the difference was ~ 10^(-3) (for last time-step). Using more timesteps has a linear effect on runtime but adding more nodes to grid (in all dimensions) has a quadratical effect on runtime.

  

