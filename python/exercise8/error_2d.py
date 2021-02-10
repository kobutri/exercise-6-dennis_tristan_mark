import pmsc
import numpy as np
import matplotlib.pyplot as plt

def error_measure_2d(communicator, nodes_per_dim, boundary_function, poisson_function):
    
    grid = pmsc.RegularGrid(communicator,pmsc.Point(0, 0), pmsc.Point(1.0, 1.0), 
                                pmsc.MultiIndex(nodes_per_dim, nodes_per_dim))
    (matrix, vector) = pmsc.assemble_poisson_matrix(grid, poisson_function, boundary_function) 
    x = pmsc.Vector(vector)

    solver = pmsc.CgSolver()
    solver.set_operator(matrix)
    solver.relative_tolerance(1e-15)
    solver.max_iterations(1000)
    solver.absolute_tolerance(0.0)
    solver.setup()
    solver.solve(x, vector)
    gridFunc = pmsc.GridFunction(grid, x)
    
    return pmsc.compute_l_infinity_error(grid, gridFunc, boundary_function)

    
def boundary_function(x):
    return np.sin(np.pi* x[0]) * np.sin(np.pi* x[1]) 

def poisson_function(x):
    return 2* np.pi**2 * np.sin(np.pi* x[0]) * np.sin(np.pi* x[1]) 


pmsc.MPI_Init()

communicator = pmsc.mpi_comm_world()

x = []
y = []
 
for i in range(2,7):                
    y.append(error_measure_2d(communicator, 2**i, boundary_function, poisson_function))
    x.append(2**(i*-1))
    

def forward(x):
    return x**(1/2)
def inverse(x):
    return x**2


plt.plot(x,y) 
plt.xlabel("h", family='serif', color='r',weight='normal', size = 16,labelpad = 6)
plt.ylabel("l_infinity_error", family='serif', color='b',weight='normal', size = 16,labelpad = 6)
plt.yscale('function', functions=(forward,inverse))
plt.show()

pmsc.MPI_Finalize()

