# Exercise-5

Over all timesteps and grid-sizes we tried onto grid size 5x5 and 5 timesteps the error is really low. The 5x5 grid with 5 timesteps gave an error of less than 2*10^(-14) for every point in the last timestep. The 30x30 grid with 30 timesteps gave an error less than 10^(-11).
So the tests with smaller grids had an smaller error than the tests with bigger grids. That could be due to the fact that the iteration-limit of solver was set to 1000 in all cases but a greater grid-size cause a poisson-matrix with more rows and columns which bring more iteration-steps for the same accuracy.
Furthermore using a 5x5 grid and more timesteps should have no negative effect on accuracy because the poisson-matrix is not getting bigger.


