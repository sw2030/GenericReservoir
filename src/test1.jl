include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
cum = Reservoir_Model((Nx, Ny, Nz), CuArray(q_oil), CuArray(q_water), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), p_ref, C_r, ϕ_ref, CuArray(ϕ), k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
cug = CuArray(g);
GenericReservoir.Solve_adaptive(cum, 0.0, 0.001, cug, 1); ## Compilation
@time R = GenericReservoir.Solve_adaptive(cum, 0.0, 0.005, cug, 20);

;;;;

