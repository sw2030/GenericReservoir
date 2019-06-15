include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
model_gpu = Reservoir_Model((Nx, Ny, Nz), CuArray(q_o), CuArray(q_w), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), CuArray(ϕ), k_r_w, k_r_o, p_cow, ρ_w, ρ_o, μ_w, μ_o);
g_gpu = CuArray(g);
println("===================Test run for Compiling===================")
GenericReservoir.Solve_SPE10(model_gpu, 0.0, 0.001, g_gpu, 1); ## Compilation
println("============================================================")

function savedata(d, fname, Num)
    h5write(string(fname, ".h5"), string("Data/",Num), Array(d[1]))
    h5write(string(fname, ".h5"), string("Log/",Num), d[2])
    nothing
end
function loaddata(fname, Num)
    d = h5read(string(fname, ".h5"), string("Data/",Num))
    logd = h5read(string(fname, ".h5"), string("Log/",Num))
    return CuArray(d), logd
end
