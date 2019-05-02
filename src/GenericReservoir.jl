module GenericReservoir

#include("Grid.jl")
include("res_jac.jl")
include("gmres.jl")
include("preconditioning.jl")
include("solvers.jl")
#include("CuStaticArrays.jl")

export set_grid, Reservoir_Model, gmres

end
