include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
cum = Reservoir_Model((Nx, Ny, Nz), CuArray(q_oil), CuArray(q_water), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), p_ref, C_r, ϕ_ref, CuArray(ϕ), k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
cug_guess = CuArray(g_guess);
cur = GenericReservoir.getresidual(cum, 0.005, cug_guess, cug_guess);

using LinearAlgebra
println(norm(cur))

