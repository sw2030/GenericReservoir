module GenericReservoir


include("res_jac.jl")
include("gmres.jl")
include("preconditioning.jl")
include("solver.jl")


export Reservoir_Model, gmres

end
