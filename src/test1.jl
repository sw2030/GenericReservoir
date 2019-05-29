include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
cum = Reservoir_Model((Nx, Ny, Nz), CuArray(q_oil), CuArray(q_water), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), p_ref, C_r, ϕ_ref, CuArray(ϕ), k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
cug = CuArray(g);
GenericReservoir.Solve_adaptive(cum, 0.0, 0.001, cug, 1); ## Compilation
@time R = GenericReservoir.Solve_adaptive(cum, 0.0, 0.005, cug, 20);

;;;;


include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
cum = Reservoir_Model((Nx, Ny, Nz), CuArray(q_oil), CuArray(q_water), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), p_ref, C_r, ϕ_ref, CuArray(ϕ), k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
R1 = loaddata("problem", 4)[1];
R2 = loaddata("problem", 2)[1];
R3 = loaddata("problem", 3)[1];

r1 = Array(GenericReservoir.getresidual(cum, 1.28, R1, R1));
r2 = Array(GenericReservoir.getresidual(cum, 1.28, R2, R2));
r3 = Array(GenericReservoir.getresidual(cum, 1.28, R3, R3));
