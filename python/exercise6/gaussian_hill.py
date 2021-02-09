import pmsc as mymodule
import numpy as np

def boundary_function(x, t):
    return 0

def gauss_function(x, t):
    return 0

def start_function(x):
    return 2*np.exp(-x[0]**2-2*x[1]**2)

mymodule.MPI_Init()

comm_world = mymodule.mpi_comm_world()

grid = mymodule.RegularGrid(comm_world, mymodule.Point(-2.0, -2.0), mymodule.Point(2.0, 2.0), mymodule.MultiIndex(28, 28))

phi_0 = mymodule.GridFunction(grid, start_function)

mymodule.write_to_vtk('./gaussian_hill_30_timesteps_28_grid_' + str(0), phi_0, 'gaussian_hill'  )

t = 2/60 

delta_t = 2/60

solver = mymodule.CgSolver()

solver.relative_tolerance(1e-15)

solver.max_iterations(1000)

solver.absolute_tolerance(0.0)

for i in range(1,31):  
    
    (matrix_, vector) = mymodule.assemble_heat_matrix(grid, phi_0, t, delta_t, gauss_function, boundary_function) 
    
    t+= 2/60 
    
    x = mymodule.Vector(vector)

    solver.set_operator(matrix_)

    solver.setup()

    solver.solve(x, vector)
    
    gridFunc = mymodule.GridFunction(grid, x)

    mymodule.write_to_vtk('./gaussian_hill_30_timesteps_28_grid_' + str(i), gridFunc, 'gaussian_hill' )
    
    phi_0 = gridFunc
        
    
mymodule.MPI_Finalize()