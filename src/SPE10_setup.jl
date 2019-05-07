using LinearAlgebra, StaticArrays
using HDF5

porosity = h5read("spe10data.h5", "data/porosity")
kx = h5read("spe10data.h5", "data/kx")
ky = h5read("spe10data.h5", "data/ky")
kz = h5read("spe10data.h5", "data/kz")
logradius = h5read("logdata.h5", "data/logrerw")
kraw = [@SVector([kx[i,j,k], ky[i,j,k], kz[i,j,k]]) for i in 1:60, j in 1:220, k in 1:85];
Nx, Ny, Nz = 60, 220, 85;
p_ref, ϕ_ref = 14.7, 0.2                           ## ϕ_ref is unused, p_ref used for ρ
S_wc, S_or = 0.2, 0.2                              ## SPE10 Config
k_r_w(x)   = ((x-0.2)/(1-0.2-0.2))^2#((x-S_wc)/(1-S_wc-S_or))^2            ## SPE10 Config 
k_r_o(x)   = (1-(x-0.2)/(1-0.2-0.2))^2#(1 - (x-S_wc)/(1-S_wc-S_or))^2        ## SPE10 Config ## Negligible Capillary Pressure? 
p_cow(x)   = 0.0                                   ## 6.3/log(0.00001)*log(x + 0.00001) 
C_water, C_r, C_oil    = 3e-6, 1e-6, 1.4e-6        ## C_w, C_r given SPE10 Config, C_oil?
ρ_water(p) = 64.0*exp(3e-6*(p-14.7))#64.0*exp(C_water*(p-p_ref))           ## Anyway \approx 64.0 - given in SPE10
ρ_oil(p)   = 53.0*exp(1.4e-6*(p-14.7))#53.0*exp(C_oil*(p-p_ref))             ## Anyway \approx 53.0 - given in SPE10
μ_water, μ_oil = 0.3, 3.0 # cp ### SPE10 Config gives water viscosity, also oil pvt table gives \approx 3
## 3d model
## Porosity proportional control (PI propotional)

Lx, Ly, Lz = 1200, 2200, 170
Δx = fill(Lx/Nx, Nx, Ny, Nz)
Δy = fill(Ly/Ny, Nx, Ny, Nz)
Δz = fill(Lz/Nz, Nx, Ny, Nz)
z = [12000.0+2.0*k-1.0 for i in 1:60, j in 1:220, k in 1:85]
# Top layer(k=1) z is 12001, end 12169
ϕ = copy(porosity)
kx_pad, ky_pad, kz_pad = zeros(Nx+2, Ny+2, Nz+2), zeros(Nx+2, Ny+2, Nz+2), zeros(Nx+2, Ny+2, Nz+2)
kx_pad[2:end-1, 2:end-1, 2:end-1] .= kx
ky_pad[2:end-1, 2:end-1, 2:end-1] .= ky
kz_pad[2:end-1, 2:end-1, 2:end-1] .= kz
k = (kx_pad, ky_pad, kz_pad)

Total = 5000.0
q_oil   = zeros(Nx, Ny, Nz)
q_water = zeros(Nx, Ny, Nz)
for i in (1,Nx), j in (1,Ny), kk in 1:Nz
    q_oil[i,j,kk] = 1.0;
end
halfx, halfy = round(Int, Nx/2),round(Int, Ny/2)
for i in 1:Nz ### Injector
    q_water[halfx,halfy,i] = -Total*(kx[halfx,halfy,i]/sum(kx[halfx,halfy,:]))
end 
m = Reservoir_Model((Nx, Ny, Nz), q_oil, q_water, (Δx, Δy, Δz), z, k, logradius, p_ref, C_r, ϕ_ref, ϕ, 
                k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil); 


g = [i%2==0 ? 0.2 : 6000.0 for i in 1:2*Nx*Ny*Nz];

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

;;
