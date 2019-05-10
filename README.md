# GenericReservoir

Reservoir Simulator works on Generic Number types

## Package Install
```
]add CuArray
]add CUDAnative
]add StaticArrays
]add ForwardDiff
]add HDF5
]add https://github.com/ranjanan/DIA.jl.git
```


## SPE10 Simulation 
### Setup
```
include("GenericReservoir.jl");
using Main.GenericReservoir, CuArrays
include("SPE10_setup.jl");
cum = Reservoir_Model((Nx, Ny, Nz), CuArray(q_oil), CuArray(q_water), (CuArray(Δx), CuArray(Δy), CuArray(Δz)), CuArray(z), (CuArray(kx_pad), CuArray(ky_pad), CuArray(kz_pad)), CuArray(logradius), p_ref, C_r, ϕ_ref, CuArray(ϕ), k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
cug = CuArray(g);
GenericReservoir.Solve_adaptive(cum, 0.0, 0.001, cug, 1); ## Compilation
```
### Simulation
```
steps = 20
initial_dt = 0.005
initial_t  = 0.0
GenericReservoir.Solve_adaptive(cum, initial_t, initial_dt, cug, steps);
```
