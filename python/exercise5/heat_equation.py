import pmsc as mymodule

def boundary_function(x, t):
    return x[0]**2 + 2 * x[1]**2 + 1 + 2 * t

def poisson_function(x, t):
    return -4

def start_function(x):
    return x[0]**2 + 2 * x[1]**2 + 1

mymodule.MPI_Init()

comm_world = mymodule.mpi_comm_world()

grid = mymodule.RegularGrid(comm_world, mymodule.Point(0, 0), mymodule.Point(1.0, 1.0), mymodule.MultiIndex(30, 30))

phi_0 = mymodule.GridFunction(grid, start_function)

mymodule.write_to_vtk('./output/heat_timesteps_' + str(0), phi_0, 'heat_timesteps_'  )

t = 2/10 

delta_t = 2/10 

solver = mymodule.CgSolver()

solver.relative_tolerance(1e-15)

solver.max_iterations(1000)

solver.absolute_tolerance(0.0)

for i in range(1,11):  
    
    (matrix_, vector) = mymodule.assemble_heat_matrix(grid, phi_0, t, delta_t, poisson_function, boundary_function) 
    
    t+= 2/10 
    
    x = mymodule.Vector(vector)

    solver.set_operator(matrix_)

    solver.setup()

    solver.solve(x, vector)
	
    gridFunc = mymodule.GridFunction(grid, x)

    mymodule.write_to_vtk('./output/heat_timesteps_' + str(i), gridFunc, 'heat_timesteps_' )
    
    phi_0 = gridFunc
        
    
mymodule.MPI_Finalize()