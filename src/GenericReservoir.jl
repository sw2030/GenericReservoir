module GenericReservoir

using LinearAlgebra
using CUDAnative
using HDF5
using CuArrays
using StaticArrays
using ForwardDiff
using DIA

include("res_jac.jl")
include("gmres.jl")
include("preconditioning.jl")
include("solver.jl")
include("tridiagonal.jl")
include("spe10.jl")
include("util.jl")


export gmres, testsolve, ReservoirSolve, spe10_gpu, spe10_cpu

end
