import pmsc as mymodule


def boundary_function(x):
    return x[0]**2 + 2 * x[1]**2 + 1

def poisson_function(x):
    return -6


mymodule.MPI_Init()

grid = mymodule.RegularGrid(mymodule.mpi_comm_world(), mymodule.Point(0, 0), mymodule.Point(1.0, 1.0), mymodule.MultiIndex(16, 16))

(matrix_, vector) = mymodule.assemble_poisson_matrix(grid, poisson_function, boundary_function) 

x = mymodule.Vector(vector)

solver = mymodule.CgSolver()

solver.set_operator(matrix_)

solver.relative_tolerance(1e-15)

solver.max_iterations(1000)

solver.absolute_tolerance(0.0)

solver.setup()

solver.solve(x, vector)

gridFunc =mymodule.GridFunction(grid, x)

mymodule.write_to_vtk('./output/poisson_func_2', gridFunc, 'poisson_func')
        
    
mymodule.MPI_Finalize()