using LinearAlgebra, CUDAnative
using HDF5


ϕ = h5read("spe10data.h5", "data/porosity")
kx = h5read("spe10data.h5", "data/kx")
ky = h5read("spe10data.h5", "data/ky")
kz = h5read("spe10data.h5", "data/kz")
logradius = h5read("spe10data.h5", "data/logrerw")

Nx, Ny, Nz = 60, 220, 85;
k_r_w(x)   = ((x-0.2)/(1-0.2-0.2))^2     #((x-S_wc)/(1-S_wc-S_or))^2            ## SPE10 Config 
k_r_o(x)   = (1-(x-0.2)/(1-0.2-0.2))^2   #(1 - (x-S_wc)/(1-S_wc-S_or))^2        ## SPE10 Config ## Negligible Capillary Pressure? 
p_cow(x)   = 0.0                                   ## 6.3/log(0.00001)*log(x + 0.00001) 
ρ_w(p)   = 64.0*CUDAnative.exp(3e-6*(p-6000.0))   #64.0*exp(C_water*(p-p_ref))           ## Anyway \approx 64.0 - given in SPE10
ρ_o(p)   = 53.0*CUDAnative.exp(1.4e-6*(p-6000.0)) #53.0*exp(C_oil*(p-p_ref))             ## Anyway \approx 53.0 - given in SPE10
V_mul(p) = CUDAnative.exp(1e-6*(p-6000))
μ_w, μ_o = 0.3, 3.0 # cp ### SPE10 Config gives water viscosity, also oil pvt table gives \approx 3

Lx, Ly, Lz = 1200, 2200, 170
Δx, Δy, Δz = fill(Lx/Nx, Nx, Ny, Nz), fill(Ly/Ny, Nx, Ny, Nz), fill(Lz/Nz, Nx, Ny, Nz)
z = [12000.0+2*k-1-12001 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
# Top layer(k=1) z is 12001, end 12169
kx_pad, ky_pad, kz_pad = zeros(Nx+2, Ny+2, Nz+2), zeros(Nx+2, Ny+2, Nz+2), zeros(Nx+2, Ny+2, Nz+2)
kx_pad[2:end-1, 2:end-1, 2:end-1] .= kx
ky_pad[2:end-1, 2:end-1, 2:end-1] .= ky
kz_pad[2:end-1, 2:end-1, 2:end-1] .= kz
k = (kx_pad, ky_pad, kz_pad)

q_o = zeros(Nx, Ny, Nz)
q_w = zeros(Nx, Ny, Nz)
q_o[[1,Nx],[1,Ny],:] .= 1.0;
q_w[30,110,:] .= -5000.0*(kx[30,110,:]/sum(kx[30,110,:]))

ρ_w_cpu(p)   = 64.0*exp(3e-6*(p-6000.0))
ρ_o_cpu(p)   = 53.0*exp(1.4e-6*(p-6000.0))
V_mul_cpu(p)     = exp(1e-6*(p-6000.0))

m = Reservoir_Model((Nx, Ny, Nz), q_o, q_w, (Δx, Δy, Δz), z, k, logradius, ϕ, k_r_w, k_r_o, p_cow, ρ_w_cpu, ρ_o_cpu, μ_w, μ_o, V_mul_cpu); 
g = [i%2==0 ? 0.2 : 6000.0 for i in 1:2*Nx*Ny*Nz];


;
