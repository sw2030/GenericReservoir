using ForwardDiff, DIA, CuArrays, CUDAnative


## General Node convention - for coordinate i, j, k
## nd = (k-1) * Nx * Ny + (j-1) * Nx + i 

struct Reservoir_Model{T, Tg<:AbstractArray}
    dim::NTuple{3,Int}
    q_oil::Tg
    q_water::Tg
    Δ::NTuple{3,Tg}
    z::Tg
    k::NTuple{3,Tg}
    logr::Tg
    p_ref::T
    C_r::T
    ϕ_ref::T
    ϕ::Tg
    k_r_w::Function
    k_r_o::Function
    p_cow::Function
    C_w::T
    C_o::T
    ρ_w::Function
    ρ_o::Function
    μ_w::T
    μ_o::T
end
Base.size(M::Reservoir_Model) = M.dim

## k is Nx+2, Ny+2, Nz+2 sized Array. It contains boundary(zero) information - to get i, j, k cell permeability, k[i+1, j+1, k+1] needed
function _residual_cell(g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g_prev1, g_prev2, i, j, k, mdim, mq_oil, mq_water, mΔ, mz, mk, mlogr, mp_ref, mC_r, mϕ_ref, mϕ, mk_r_w, mk_r_o, mp_cow, mC_w, mC_o, mρ_w, mρ_o, mμ_w, mμ_o, Δt)#;bth=true, p_bth=4000.0, maxinj = 10000.0)
	
    maxinj=10000.0
    bth = true
    p_bth = 4000.0
    
    Nx, Ny, Nz = mdim

    # Define nodes of neighbors
    im1 = max(1, i-1)
    ip1 = min(Nx, i+1)
    jm1 = max(1, j-1)
    jp1 = min(Ny, j+1)
    km1 = max(1, k-1)
    kp1 = min(Nz, k+1)
    
    injection = g7
    
    # Injection
    q_w = injection>maxinj ? 0.0 : mq_water[i,j,k]  ## if pressure is larger than max, no injection
    
    ###---------------------------------------------------------------------
    ### Compute inter-block quantities
    ###---------------------------------------------------------------------   
    Δx_west  = (mΔ[1][im1,j,k] + mΔ[1][i,j,k])/2
    Δx_east  = (mΔ[1][ip1,j,k] + mΔ[1][i,j,k])/2
    Δy_south = (mΔ[2][i,jm1,k] + mΔ[2][i,j,k])/2
    Δy_north = (mΔ[2][i,jp1,k] + mΔ[2][i,j,k])/2
    Δz_below = (mΔ[3][i,j,km1] + mΔ[3][i,j,k])/2
    Δz_above = (mΔ[3][i,j,kp1] + mΔ[3][i,j,k])/2
    
    # The zy area between two blocks is calculated as the arithmetic mean of the zy area at each block center
    # See note 3
    A_west  = (mΔ[3][im1,j,k]*mΔ[2][im1,j,k] + mΔ[3][i,j,k]*mΔ[2][i,j,k])/2
    A_east  = (mΔ[3][ip1,j,k]*mΔ[2][ip1,j,k] + mΔ[3][i,j,k]*mΔ[2][i,j,k])/2
    A_south = (mΔ[3][i,jm1,k]*mΔ[1][i,jm1,k] + mΔ[3][i,j,k]*mΔ[1][i,j,k])/2
    A_north = (mΔ[3][i,jp1,k]*mΔ[1][i,jp1,k] + mΔ[3][i,j,k]*mΔ[1][i,j,k])/2
    A_above = (mΔ[1][i,j,km1]*mΔ[2][i,j,km1] + mΔ[1][i,j,k]*mΔ[2][i,j,k])/2
    A_below = (mΔ[1][i,j,kp1]*mΔ[2][i,j,kp1] + mΔ[1][i,j,k]*mΔ[2][i,j,k])/2

    # The interface permeability is the harmonic average of the two grid blocks
    # k is the controling factor of zeros - if neighbor gets out of bounds it becomes 0
    # Need k as a Grid
    # See note 3
    k_west  = (mΔ[1][im1,j,k]+mΔ[1][i,j,k])*mk[1][i,j+1,k+1]  *mk[1][i+1,j+1,k+1]/(mΔ[1][im1,j,k]*mk[1][i+1,j+1,k+1] + mΔ[1][i,j,k]*mk[1][i,j+1,k+1])
    k_east  = (mΔ[1][ip1,j,k]+mΔ[1][i,j,k])*mk[1][i+2,j+1,k+1]*mk[1][i+1,j+1,k+1]/(mΔ[1][ip1,j,k]*mk[1][i+1,j+1,k+1] + mΔ[1][i,j,k]*mk[1][i+2,j+1,k+1])
    k_south = (mΔ[2][i,jm1,k]+mΔ[2][i,j,k])*mk[2][i+1,j,k+1]  *mk[2][i+1,j+1,k+1]/(mΔ[2][i,jm1,k]*mk[2][i+1,j+1,k+1] + mΔ[2][i,j,k]*mk[2][i+1,j,k+1])
    k_north = (mΔ[2][i,jp1,k]+mΔ[2][i,j,k])*mk[2][i+1,j+2,k+1]*mk[2][i+1,j+1,k+1]/(mΔ[2][i,jp1,k]*mk[2][i+1,j+1,k+1] + mΔ[2][i,j,k]*mk[2][i+1,j+2,k+1])
    k_below = (mΔ[3][i,j,km1]+mΔ[3][i,j,k])*mk[3][i+1,j+1,k]  *mk[3][i+1,j+1,k+1]/(mΔ[3][i,j,km1]*mk[3][i+1,j+1,k+1] + mΔ[3][i,j,k]*mk[3][i+1,j+1,k])
    k_above = (mΔ[3][i,j,kp1]+mΔ[3][i,j,k])*mk[3][i+1,j+1,k+2]*mk[3][i+1,j+1,k+1]/(mΔ[3][i,j,kp1]*mk[3][i+1,j+1,k+1] + mΔ[3][i,j,k]*mk[3][i+1,j+1,k+2])
    

    ###---------------------------------------------------------------------
    ### Load S, p and use Aux equation
    ###---------------------------------------------------------------------   
    #Saturation load
    S_w_im1jk   = g2
    S_w_ijm1k   = g4
    S_w_ijkm1   = g6
    S_w_ijk     = g8
    S_w_ijkp1   = g10
    S_w_ijp1k   = g12
    S_w_ip1jk   = g14
    S_w_prev    = g_prev2

    # Using Aux Eq
    S_o_im1jk   = 1-S_w_im1jk
    S_o_ijm1k   = 1-S_w_ijm1k
    S_o_ijkm1   = 1-S_w_ijkm1
    S_o_ijk     = 1-S_w_ijk
    S_o_ijkp1   = 1-S_w_ijkp1
    S_o_ijp1k   = 1-S_w_ijp1k
    S_o_ip1jk   = 1-S_w_ip1jk
    S_o_prev    = 1-S_w_prev

    # Pressure load
    p_o_im1jk   = g1
    p_o_ijm1k   = g3
    p_o_ijkm1   = g5
    p_o_ijk     = g7
    p_o_ijkp1   = g9
    p_o_ijp1k   = g11
    p_o_ip1jk   = g13
    p_o_prev   = g_prev1

    # Capillary pressure
    p_w_im1jk = p_o_im1jk - mp_cow(S_w_im1jk)
    p_w_ijm1k = p_o_ijm1k - mp_cow(S_w_ijm1k)
    p_w_ijkm1 = p_o_ijkm1 - mp_cow(S_w_ijkm1)
    p_w_ijk   = p_o_ijk   - mp_cow(S_w_ijk)
    p_w_ijkp1 = p_o_ijkp1 - mp_cow(S_w_ijkp1)
    p_w_ijp1k = p_o_ijp1k - mp_cow(S_w_ijp1k)
    p_w_ip1jk = p_o_ip1jk - mp_cow(S_w_ip1jk)
    p_w_prev  = p_o_prev  - mp_cow(S_w_prev)
    
    # 5.615 is oil field units correction factor. See Note 3.
    V_ijk      = mΔ[1][i,j,k]*mΔ[2][i,j,k]*mΔ[3][i,j,k]*mϕ[i,j,k]/5.615
    V_ijk_prev = mΔ[1][i,j,k]*mΔ[2][i,j,k]*mΔ[3][i,j,k]*mϕ[i,j,k]/5.615
    
    
    ###---------------------------------------------------------------------
    ### Calculate Fluid potentials Φ
    ###---------------------------------------------------------------------
    Φ_w_im1jk = p_w_im1jk - mρ_w(p_w_im1jk)*mz[im1,j,k]/144.0
    Φ_w_ijm1k = p_w_ijm1k - mρ_w(p_w_ijm1k)*mz[i,jm1,k]/144.0
    Φ_w_ijkm1 = p_w_ijkm1 - mρ_w(p_w_ijkm1)*mz[i,j,km1]/144.0
    Φ_w_ijk   = p_w_ijk   - mρ_w(p_w_ijk)  *mz[i,j,k]  /144.0
    Φ_w_ijkp1 = p_w_ijkp1 - mρ_w(p_w_ijkp1)*mz[i,j,kp1]/144.0
    Φ_w_ijp1k = p_w_ijp1k - mρ_w(p_w_ijp1k)*mz[i,jp1,k]/144.0
    Φ_w_ip1jk = p_w_ip1jk - mρ_w(p_w_ip1jk)*mz[ip1,j,k]/144.0

    Φ_o_im1jk = p_o_im1jk - mρ_o(p_o_im1jk)*mz[im1,j,k]/144.0
    Φ_o_ijm1k = p_o_ijm1k - mρ_o(p_o_ijm1k)*mz[i,jm1,k]/144.0
    Φ_o_ijkm1 = p_o_ijkm1 - mρ_o(p_o_ijkm1)*mz[i,j,km1]/144.0
    Φ_o_ijk   = p_o_ijk   - mρ_o(p_o_ijk)  *mz[i,j,k]  /144.0
    Φ_o_ijkp1 = p_o_ijkp1 - mρ_o(p_o_ijkp1)*mz[i,j,kp1]/144.0
    Φ_o_ijp1k = p_o_ijp1k - mρ_o(p_o_ijp1k)*mz[i,jp1,k]/144.0
    Φ_o_ip1jk = p_o_ip1jk - mρ_o(p_o_ip1jk)*mz[ip1,j,k]/144.0
	
    
    ###---------------------------------------------------------------------
    ### Compute Relative Permeability
    ###---------------------------------------------------------------------
    # Upstream condition. Relative permeability is always a function of S_water!    
    k_r_w_west  = Φ_w_im1jk > Φ_w_ijk ? mk_r_w(S_w_im1jk)*mρ_w(p_w_im1jk) : mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)
    k_r_w_east  = Φ_w_ip1jk > Φ_w_ijk ? mk_r_w(S_w_ip1jk)*mρ_w(p_w_ip1jk) : mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)
    k_r_w_south = Φ_w_ijm1k > Φ_w_ijk ? mk_r_w(S_w_ijm1k)*mρ_w(p_w_ijm1k) : mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)
    k_r_w_north = Φ_w_ijp1k > Φ_w_ijk ? mk_r_w(S_w_ijp1k)*mρ_w(p_w_ijp1k) : mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)
    k_r_w_below = Φ_w_ijkm1 > Φ_w_ijk ? mk_r_w(S_w_ijkm1)*mρ_w(p_w_ijkm1) : mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)
    k_r_w_above = Φ_w_ijkp1 > Φ_w_ijk ? mk_r_w(S_w_ijkp1)*mρ_w(p_w_ijkp1) : mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)


    k_r_o_west  = Φ_o_im1jk > Φ_o_ijk ? mk_r_o(S_w_im1jk)*mρ_o(p_o_im1jk) : mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)
    k_r_o_east  = Φ_o_ip1jk > Φ_o_ijk ? mk_r_o(S_w_ip1jk)*mρ_o(p_o_ip1jk) : mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)
    k_r_o_south = Φ_o_ijm1k > Φ_o_ijk ? mk_r_o(S_w_ijm1k)*mρ_o(p_o_ijm1k) : mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)
    k_r_o_north = Φ_o_ijp1k > Φ_o_ijk ? mk_r_o(S_w_ijp1k)*mρ_o(p_o_ijp1k) : mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)
    k_r_o_below = Φ_o_ijkm1 > Φ_o_ijk ? mk_r_o(S_w_ijkm1)*mρ_o(p_o_ijkm1) : mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)
    k_r_o_above = Φ_o_ijkp1 > Φ_o_ijk ? mk_r_o(S_w_ijkp1)*mρ_o(p_o_ijkp1) : mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)
    

    ###---------------------------------------------------------------------
    ### Calculate Interblock Transmissibility
    ###---------------------------------------------------------------------
    # The 1.127e-3 factor is oil field units. See Note 4.
    T_w_west  = 1.127e-3*k_west*k_r_w_west/mμ_w*A_west/Δx_west # boundary condition
    T_w_east  = 1.127e-3*k_east*k_r_w_east/mμ_w*A_east/Δx_east
    T_w_south = 1.127e-3*k_south*k_r_w_south/mμ_w*A_south/Δy_south
    T_w_north = 1.127e-3*k_north*k_r_w_north/mμ_w*A_north/Δy_north
    T_w_below = 1.127e-3*k_below*k_r_w_below/mμ_w*A_below/Δz_below
    T_w_above = 1.127e-3*k_above*k_r_w_above/mμ_w*A_above/Δz_above

    T_o_west    = 1.127e-3*k_west*k_r_o_west/mμ_o*A_west/Δx_west
    T_o_east    = 1.127e-3*k_east*k_r_o_east/mμ_o*A_east/Δx_east
    T_o_south   = 1.127e-3*k_south*k_r_o_south/mμ_o*A_south/Δy_south
    T_o_north   = 1.127e-3*k_north*k_r_o_north/mμ_o*A_north/Δy_north
    T_o_below   = 1.127e-3*k_below*k_r_o_below/mμ_o*A_below/Δz_below
    T_o_above   = 1.127e-3*k_above*k_r_o_above/mμ_o*A_above/Δz_above

    
    ###---------------------------------------------------------------------
    ### Impose Well Condition
    ###---------------------------------------------------------------------
    # Calculate Wellbore Radius(SPE-10 Config), Peaceman Radius r_e (Chen. 449p) -> LOOK SPE10_Setup.jl
    # Calculate Productivity index (Well-Index) if it is production well
    #
    PI = mq_oil[i,j,k] > 0 ? 7.06e-3*((mk[1][i+1,j+1,k+1]))*mΔ[3][i,j,k]/mlogr[i,j,k] : 0.0
    
    # 120001 is depth of center of the surface level grids
    Φ_diff_o   = p_o_ijk-p_bth-mρ_o(p_o_ijk)*(mz[i,j,k]-12001)/144
    Φ_diff_w   = p_w_ijk-p_bth-mρ_w(p_w_ijk)*(mz[i,j,k]-12001)/144
    well_o     = PI==0.0 ? 0.0 : PI*Φ_diff_o*mk_r_o(S_w_ijk)*mρ_o(p_o_ijk)/mμ_o
    well_w     = PI==0.0 ? 0.0 : PI*Φ_diff_w*mk_r_w(S_w_ijk)*mρ_w(p_w_ijk)/mμ_w
    
    
    ###---------------------------------------------------------------------
    ### Calculate Residuals
    ###---------------------------------------------------------------------
    residual_water_ijk = T_w_west*(Φ_w_im1jk - Φ_w_ijk)   +
                         T_w_east*(Φ_w_ip1jk - Φ_w_ijk)   +
                         T_w_south*(Φ_w_ijm1k - Φ_w_ijk)  +
                         T_w_north*(Φ_w_ijp1k - Φ_w_ijk)  +
                         T_w_above*(Φ_w_ijkp1 - Φ_w_ijk)  +
                         T_w_below*(Φ_w_ijkm1 - Φ_w_ijk)  -
                         q_w*mρ_w(p_w_ijk)                -
                         well_w                            -
                         (V_ijk*S_w_ijk*mρ_w(p_w_ijk)     -
                         V_ijk_prev*S_w_prev*mρ_w(p_w_prev))/Δt
    
    residual_oil_ijk   = T_o_west*(Φ_o_im1jk - Φ_o_ijk)   +
                         T_o_east*(Φ_o_ip1jk - Φ_o_ijk)   +
                         T_o_south*(Φ_o_ijm1k - Φ_o_ijk)  +
                         T_o_north*(Φ_o_ijp1k - Φ_o_ijk)  +
                         T_o_above*(Φ_o_ijkp1 - Φ_o_ijk)  +
                         T_o_below*(Φ_o_ijkm1 - Φ_o_ijk)  -
                         well_o                            -
                         (V_ijk*S_o_ijk*mρ_o(p_o_ijk)     -
                         V_ijk_prev*S_o_prev*mρ_o(p_o_prev))/Δt
   
			 
    return residual_water_ijk, residual_oil_ijk
    
end
_residual_cell_pre(m, Δt, g, g_prev, i, j, k) = _residual_cell(g...,  g_prev[1], g_prev[2], i, j, k, m.dim, m.q_oil, m.q_water, m.Δ, m.z, m.k, m.logr, m.p_ref, m.C_r, m.ϕ_ref, m.ϕ, m.k_r_w, m.k_r_o, m.p_cow, m.C_w, m.C_o, m.ρ_w, m.ρ_o, m.μ_w, m.μ_o, Δt)


## Residual for Regular Arrays
function getresidual(m::Reservoir_Model{T, Array{T,3}}, Δt, g::Array{T,1}, g_prev::Array{T,1}) where {T}
    Nx, Ny, Nz = size(m)
    res = similar(g)
    z = zeros(2)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        nd = (k-1) * Nx * Ny + (j-1) * Nx + i 
        input = (i==1 ? z : g[2nd-3:2nd-2], j==1 ? z : g[2nd-2Nx-1:2nd-2Nx], k==1 ? z : g[2nd-2*Nx*Ny-1:2nd-2Nx*Ny],
                  g[2nd-1:2nd], k==Nz ? z : g[2nd+2*Nx*Ny-1:2nd+2*Nx*Ny], j==Ny ? z : g[2nd+2Nx-1:2nd+2Nx], i==Nx ? z : g[2nd+1:2nd+2])
	res[2nd-1:2nd] .= _residual_cell_pre(m, Δt, [input[b][a] for a in 1:2, b in 1:7][:], g_prev[2nd-1:2nd], i, j, k)
    end
    return res
end
## Residual GPU version
function getresidual(m::Reservoir_Model{T, CuArray{T,3}}, Δt, g::CuArray{T,1}, g_prev::CuArray{T,1}) where {T}
    res = zero(g) #Later replace with similar
    #inputchunk = CuArray(fill(Tuple(zeros(14)), size(m)...))
    _getresidual_prealloc(res, m, Δt, g, g_prev)
    return res
end
function _getresidual_prealloc(res::CuArray{T,1}, m::Reservoir_Model{T, CuArray{T,3}}, Δt, g::CuArray{T,1}, g_prev::CuArray{T,1}) where {T}
    Nxx, Nyy, Nzz = size(m)
    
    ## Passing in Function _residual_cell
    function kernel(f, res, mdim, mq_oil, mq_water, mΔ, mz, mk, mlogr, mp_ref, mC_r, mϕ_ref, mϕ, mk_r_w, mk_r_o, mp_cow, mC_w, mC_o, mρ_w, mρ_o, mμ_w, mμ_o, Δt, g, g_prev)
   	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        k = (blockIdx().z-1) * blockDim().z + threadIdx().z
	
	Nx, Ny, Nz = mdim
	if i<=Nx && j<=Ny && k<=Nz
	    nd = (k-1) * Nx * Ny + (j-1) * Nx + i
	    #Since we are not allowed to allocate an input array inside CUDA kernel, we just do it elementwise here.
	    #The code below is the one we will duplicate elementwise
	    #input = [i==1 ? z : g[2nd-3:2nd-2], j==1 ? z : g[2nd-2Nx-1:2nd-2Nx], k==1 ? z : g[2nd-2*Nx*Ny-1:2nd-2Nx*Ny], g[2nd-1:2nd], k==Nz ? z : g[2nd+2*Nx*Ny-1:2nd+2*Nx*Ny], j==Ny ? z : g[2nd+2Nx-1:2nd+2Nx], i==Nx ? z : g[2nd+1:2nd+2]]
	    g1 = i==1 ? 0.0 : g[2nd-3]
	    g2 = i==1 ? 0.0 : g[2nd-2]
	    g3 = j==1 ? 0.0 : g[2nd-2Nx-1]
	    g4 = j==1 ? 0.0 : g[2nd-2Nx]
	    g5 = k==1 ? 0.0 : g[2nd-2*Nx*Ny-1]
	    g6 = k==1 ? 0.0 : g[2nd-2Nx*Ny]
	    g7 = g[2nd-1]
	    g8 = g[2nd]
	    g9 = k==Nz ? 0.0 : g[2nd+2*Nx*Ny-1]
	    g10 = k==Nz ? 0.0 : g[2nd+2*Nx*Ny]
	    g11 = j==Ny ? 0.0 : g[2nd+2Nx-1]
	    g12 = j==Ny ? 0.0 : g[2nd+2Nx]
	    g13 = i==Nx ? 0.0 : g[2nd+1]
	    g14 = i==Nx ? 0.0 : g[2nd+2]
            rw, ro = f(g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g_prev[2nd-1], g_prev[2nd], i, j, k, mdim, mq_oil, mq_water, mΔ, mz, mk, mlogr, mp_ref, mC_r, mϕ_ref, mϕ, mk_r_w, mk_r_o, mp_cow, mC_w, mC_o, mρ_w, mρ_o, mμ_w, mμ_o, Δt)  
	    res[2nd-1] = rw
	    res[2nd] = ro
    	end

	return
    end
    
    max_threads = 256
    threads_x   = min(max_threads, Nxx)
    threads_y   = min(max_threads ÷ threads_x, Nyy)
    threads_z   = min(max_threads ÷ threads_x ÷ threads_y, Nzz)
    threads     = (threads_x, threads_y, threads_z)
    blocks      = ceil.(Int, (Nxx, Nyy, Nzz) ./ threads)

    @cuda threads=threads blocks=blocks kernel(_residual_cell, res, m.dim, m.q_oil, m.q_water, m.Δ, m.z, m.k, m.logr, m.p_ref, m.C_r, m.ϕ_ref, m.ϕ, m.k_r_w, m.k_r_o, m.p_cow, m.C_w, m.C_o, m.ρ_w, m.ρ_o, m.μ_w, m.μ_o, Δt, g, g_prev)

end


function _getjacobian_array(m::Reservoir_Model{T, Array{T,3}}, Δt, g::AbstractVector, g_prev::AbstractVector) where {T}
    Nx, Ny, Nz = size(m)
    z = zeros(2)
    jA = zeros(2*Nx*Ny*Nz, 19)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        nd = (k-1) * Nx * Ny + (j-1) * Nx + i 
        input = (i==1 ? z : g[2nd-3:2nd-2], j==1 ? z : g[2nd-2Nx-1:2nd-2Nx], k==1 ? z : g[2nd-2*Nx*Ny-1:2nd-2Nx*Ny],
                  g[2nd-1:2nd], k==Nz ? z : g[2nd+2*Nx*Ny-1:2nd+2*Nx*Ny], j==Ny ? z : g[2nd+2Nx-1:2nd+2Nx], i==Nx ? z : g[2nd+1:2nd+2])
	J = ForwardDiff.jacobian(θ->[_residual_cell_pre(m, Δt, θ, g_prev[2nd-1:2nd], i, j, k)...], [input[a][b] for b in 1:2, a in 1:7][:])

	jA[2*nd, 1:2]     .= J[2,5:6]
	jA[2*nd-1, 2:3]   .= J[1,5:6]
	jA[2*nd, 4:5]     .= J[2,3:4]
	jA[2*nd-1, 5:6]   .= J[1,3:4]
	jA[2*nd, 7:8]     .= J[2,1:2]
	jA[2*nd-1, 8:9]   .= J[1,1:2]
	jA[2*nd, 9:10]    .= J[2,7:8]
	jA[2*nd-1, 10:11] .= J[1,7:8]
	jA[2*nd, 11:12]   .= J[2,13:14]
	jA[2*nd-1, 12:13] .= J[1,13:14]
	jA[2*nd, 14:15]   .= J[2,11:12]
	jA[2*nd-1, 15:16] .= J[1,11:12]
	jA[2*nd, 17:18]   .= J[2,9:10]
	jA[2*nd-1, 18:19] .= J[1,9:10]
    end
    return jA
end
function getjacobian(m::Reservoir_Model, Δt, g::AbstractVector, g_prev::AbstractVector)
    Nx, Ny, Nz = size(m)
    Nxy = Nx*Ny
    N = Nx*Ny*Nz
    jA = _getjacobian_array(m, Δt, g, g_prev)
    diagband = [-2Nxy-1, -2Nxy, -2Nxy+1, -2Nx-1, -2Nx, -2Nx+1, -3, -2, -1, 0, 1, 2, 3, 2Nx-1, 2Nx, 2Nx+1, 2Nxy-1, 2Nxy, 2Nxy+1]
    diagidx  = [(diagband[i]<0) ? (-diagband[i]+1:2N) : (1:2N-diagband[i]) for i in 1:length(diagband)]
    return SparseMatrixDIA(Tuple([diagband[i]=>jA[diagidx[i], i] for i in 1:length(diagband)]), 2N, 2N)
end
