# GenericReservoir

Reservoir Simulator works both on CPU/GPU

## Package Install
```
]add CuArrays
]add CUDAnative#v2.1.2 ## Latest CUDAnative might not work perfectly with StaticArrays.jl ldiv!
]add StaticArrays
]add ForwardDiff
]add HDF5
]add https://github.com/ranjanan/DIA.jl.git

]add https://github.com/sw2030/GenericReservoir.jl.git
```


## GPU SPE10 Simulation(In REPL) 
### Setup
```
using GenericReservoir
m, g = spe10_gpu("spe10data.h5");  ## copy data/spe10data.h5 or use your directory/filename
testsolve(m, g);
```
### Simulation
```
max_steps = 200
initial_dt = 0.005
initial_t  = 0.0

ReservoirSolve(m, initial_t, initial_dt, g, max_steps);
```
