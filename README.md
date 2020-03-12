# GenericReservoir

Reservoir Simulator works on Generic Number types

## Package Install
```
]add CuArrays
]add CUDAnative
]add StaticArrays
]add ForwardDiff
]add HDF5
]add https://github.com/ranjanan/DIA.jl.git
```


## SPE10 Simulation(In REPL) 
### Setup
```
include("test.jl")
```
### Simulation
```
max_steps = 200
initial_dt = 0.005
initial_t  = 0.00

GenericReservoir.Solve_SPE10(model_gpu, initial_t, initial_dt, g_gpu, max_steps);
```
