include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
model_gpu = Reservoir_Model((Nx, Ny, Nz), CuArray(q_oil), CuArray(q_water), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), p_ref, C_r, ϕ_ref, CuArray(ϕ), k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
g_gpu = CuArray(g);
println("===================Test run for Compiling===================")
GenericReservoir.Solve_adaptive(model_gpu, 0.0, 0.001, g_gpu, 1); ## Compilation
println("============================================================")
