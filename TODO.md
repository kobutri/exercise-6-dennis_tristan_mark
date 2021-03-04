# TODO


- use setup methods to allocate memory beforehand for solvers --- DONE
- use English everywhere --- PROBABLY DONE
- settle on one method to specify path parameters --- not applicable
- use less memory to assemble poisson matrix by using new constructor  --- DONE
- should we store the diagonal matrix element first for each row? --- DONE?
- make gridfunction hold shared_ptr instead of unique_ptr   --- DONE
- improve grid partitioning algorithm   --- HOW?
- improve exchange vector data by using this circular exchange method
- test and improve gauss_seidel preconditioning -- DONE
- put classes in module.cpp in separate headers -- DONE
- transfer assemble_heat ... to head.h --- DONE
- cleanup recursive include disaster --- DONE
- cleanup
- make nice comments
