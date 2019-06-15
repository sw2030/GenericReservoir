using DIA, LinearAlgebra, CuArrays, StaticArrays

#=
function triLU!(P, Nxy, Nz)
    if size(P, 1)÷(Nxy*Nz)==1
	block_tridiagonal_LU!(P.diags[1].second, P.diags[2].second, P.diags[3].second, Nxy, Nz)
    else 
        block_tridiagonal_LU_block!(P.diags[1].second, P.diags[2].second, P.diags[3].second, P.diags[4].second, P.diags[5].second, P.diags[6].second, P.diags[7].second, Nxy, Nz)
    end
end
function triLU_solve!(P, d, x, Nxy, Nz)
    if size(P, 1)÷(Nxy*Nz)==1
        block_tridiagonal_LU_solve!(P.diags[1].second, P.diags[2].second, P.diags[3].second, copy(d), x, Nxy, Nz)
    else
	block_tridiagonal_LU_solve_block!(P.diags[1].second, P.diags[2].second, P.diags[3].second, P.diags[4].second, P.diags[5].second, P.diags[6].second, P.diags[7].second, copy(d), x, Nxy, Nz)
    end
end
=#
function triLU!(P)
    if size(P, 1)÷(1122000)==1
	block_execute!(tridiagonal_LU!, 32, 1122000, 170, P.diags[1].second, P.diags[2].second, P.diags[3].second)
    else
        block_execute!(tridiagonal_LU_block!, 128, 1122000, 85, P.diags[1].second, P.diags[2].second, P.diags[3].second, P.diags[4].second, P.diags[5].second, P.diags[6].second, P.diags[7].second) 
    end
end
function triLU_solve!(P, d, x)
    if size(P, 1)÷(1122000)==1
        block_execute!(tridiagonal_LU_solve!, 32, 1122000, 340, P.diags[1].second, P.diags[2].second, P.diags[3].second, copy(d), x)
    else
        block_execute!(tridiagonal_LU_solve_block!, 16, 1122000, 340, P.diags[1].second, P.diags[2].second, P.diags[3].second, P.diags[4].second, P.diags[5].second, P.diags[6].second, P.diags[7].second, copy(d), x)
    end
end

function block_execute!(f, numthreads, N, Nz, args...)
    function kernel(func, Nxy, Nz, args...)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
     	    func(args..., Nz * (i-1) + 1, i * Nz)
        end
	return
    end
    Nxy = N ÷ Nz
    @cuda threads=numthreads blocks=ceil(Int, Nxy/numthreads) kernel(f, Nxy, Nz, args...)
end

### LU tridiagonal with separate steps (Non block DIAs)
function tridiagonal_LU!(a::AbstractVector{T}, b::AbstractVector{T}, c::AbstractVector{T}, startidx, endidx) where {T}
    @inbounds for i in startidx+1:endidx
        a[i-1] /= b[i-1]
        b[i]   -= a[i-1]*c[i-1]
    end
end
function block_tridiagonal_LU!(a::CuVector{T}, b::CuVector{T}, c::CuVector{T}, Nxy, Nz) where {T}
    function kernel(f, a, b, c, Nxy, Nz)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
            f(a, b, c, Nz*(i-1) + 1, i*Nz)
        end
        return
    end
    @cuda threads=16 blocks=ceil(Int, Nxy/16) kernel(tridiagonal_LU!, a, b, c, Nxy, Nz)
end
function tridiagonal_LU_solve!(a::AbstractVector{T}, b::AbstractVector{T}, c::AbstractVector{T}, d::AbstractVector{T}, x::AbstractVector{T}, startidx, endidx) where {T}
    @inbounds for i in startidx+1:endidx
        d[i] = d[i] - a[i-1]*d[i-1]
    end
    x[endidx] = d[endidx]/b[endidx]
    @inbounds for i in (endidx-1):-1:startidx
        x[i] = (d[i] - c[i]*x[i+1])/b[i]
    end
end
function block_tridiagonal_LU_solve!(a::CuVector{T}, b::CuVector{T}, c::CuVector{T}, d::CuVector{T}, x::CuVector{T}, Nxy, Nz) where {T}
    function kernel(f, a, b, c, d, x, Nxy, Nz)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
            f(a, b, c, d, x, (i-1)*Nz + 1, i*Nz)
        end
        return
    end
    @cuda threads=16 blocks=ceil(Int, Nxy/16) kernel(tridiagonal_LU_solve!, a, b, c, d, x, Nxy, Nz)
end

### LU tridiagonal with separate steps (Block DIAS)
function tridiagonal_LU_block!(a1, a2, b1, b2, b3, c1, c2, startidx, endidx)
    @inbounds for i in startidx+1:endidx
        BTinvAT =  SMatrix{2,2}(b2[2i-3], b3[2i-3], b1[2i-3], b2[2i-2])\SMatrix{2,2}(a2[2i-3], b1[2i-2], a1[2i-3], a2[2i-2])
        a2[2i-3] = BTinvAT[1]
        b1[2i-2] = BTinvAT[2]
        a1[2i-3] = BTinvAT[3]
        a2[2i-2] = BTinvAT[4]
        LCT = SMatrix{2,2}(c1[2i-3], c2[2i-3], b3[2i-2], c1[2i-2])*BTinvAT
	b2[2i-1] = b2[2i-1] - LCT[1]
	b3[2i-1] = b3[2i-1] - LCT[2]
	b1[2i-1] = b1[2i-1] - LCT[3]
	b2[2i]   = b2[2i] - LCT[4]
    end
end
function block_tridiagonal_LU_block!(a1, a2, b1, b2, b3, c1, c2, Nxy, Nz)
    function kernel(f, a1, a2, b1, b2, b3, c1, c2, Nxy, Nz)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
            f(a1, a2, b1, b2, b3, c1, c2, (i-1)*Nz + 1, i*Nz)
        end
        return
    end
    @cuda threads=32 blocks=ceil(Int, Nxy/32) kernel(tridiagonal_LU_block!, a1, a2, b1, b2, b3, c1, c2, Nxy, Nz)
end
function tridiagonal_LU_solve_block!(a1, a2, b1, b2, b3, c1, c2, d, x, startidx, endidx)
    for i in startidx+1:endidx
        AD = SMatrix{2,2}(a2[2i-3], a1[2i-3], b1[2i-2], a2[2i-2])*SVector{2}(d[2i-3], d[2i-2])
        d[2i-1] -= AD[1]
        d[2i]   -= AD[2]
    end
    xendidx = SMatrix{2,2}(b2[2endidx-1], b1[2endidx-1], b3[2endidx-1], b2[2endidx])\SVector{2}(d[2endidx-1], d[2endidx])
    x[2endidx-1] = xendidx[1]
    x[2endidx]   = xendidx[2]
    for i in (endidx-1):-1:startidx
        binvdminuscx = SMatrix{2,2}(b2[2i-1], b1[2i-1], b3[2i-1], b2[2i])\(SVector{2}(d[2i-1], d[2i]) - SMatrix{2,2}(c1[2i-1], b3[2i], c2[2i-1], c1[2i]) * SVector{2}(x[2i+1], x[2i+2]))
        x[2i-1] = binvdminuscx[1]
        x[2i]   = binvdminuscx[2]
    end
end
function block_tridiagonal_LU_solve_block!(a1, a2, b1, b2, b3, c1, c2, d, x, Nxy, Nz)
    function kernel(f, a1, a2, b1, b2, b3, c1, c2, d, x, Nxy, Nz)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
            f(a1, a2, b1, b2, b3, c1, c2, d, x, (i-1)*Nz + 1, i * Nz)
        end
        return
    end
    @cuda threads=32 blocks=ceil(Int, Nxy/32) kernel(tridiagonal_LU_solve_block!, a1, a2, b1, b2, b3, c1, c2, d, x, Nxy, Nz)
end


### Block level tridiagonal solve
function tridiagonal_solve!(a::AbstractVector{T}, b::AbstractVector{T}, c::AbstractVector{T}, d::AbstractVector{T}, x::AbstractVector{T}, startidx, endidx) where {T} # b, d, x from startidx:endidx, a, c from startidx:endidx-1
    @inbounds for i in startidx+1:endidx
        b[i] = b[i] - a[i-1]*c[i-1]/b[i-1]
        d[i] = d[i] - a[i-1]*d[i-1]/b[i-1]
    end
    x[endidx] = d[endidx]/b[endidx]
    @inbounds for i in (endidx-1):-1:startidx
        x[i] = (d[i] - c[i]*x[i+1])/b[i]
    end
end
function tridiagonal_solve!(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, startidx, endidx) # a, b, c 4 Vectors each, d, x are gonna be long vector
    @inbounds for i in startidx:endidx-1
        diff = SMatrix{2,2}(a1[i], a3[i], a2[i], a4[i])*((SMatrix{2,2}(b1[i], b3[i], b2[i], b4[i]))\SMatrix{2,3}(c1[i], c3[i], c2[i], c4[i], d[2i-1], d[2i]))  # Diff = A[i-1] * (B[i-1] \ [C[i-1] | d[i-1]])
        b1[i+1]  -= diff[1]
        b2[i+1]  -= diff[3]
        b3[i+1]  -= diff[2]
        b4[i+1]  -= diff[4]
        d[2i+1] -= diff[5]
        d[2i+2]   -= diff[6]
    end
    xendidx = SMatrix{2,2}(b1[endidx], b3[endidx], b2[endidx], b4[endidx])\SVector{2}(d[2endidx-1], d[2endidx])
    x[2endidx-1] = xendidx[1]
    x[2endidx] = xendidx[2]
    @inbounds for i in (endidx-1):-1:startidx
        xi = (SMatrix{2,2}(b1[i], b3[i], b2[i], b4[i]))\(SVector{2}(d[2i-1], d[2i]) - SMatrix{2,2}(c1[i], c3[i], c2[i], c4[i]) * SVector{2}(x[2i+1], x[2i+2]))
        x[2i-1] = xi[1]
        x[2i] = xi[2]
    end
end

### CUDAnative implementation of block tridiagonal
function block_tridiagonal_solve!(a::CuVector{T}, b::CuVector{T}, c::CuVector{T}, d::CuVector{T}, x::CuVector{T}, Nxy, Nz) where {T} ## Assume Input is Nx*Ny*Nz length vectors(b, d, x)
    function kernel(f, a, b, c, d, x, Nxy, Nz)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
            f(a, b, c, d, x, (i-1) * Nz + 1, i * Nz)
        end
        return
    end
    @cuda threads=16 blocks=ceil(Int, Nxy/16) kernel(tridiagonal_solve!, a, b, c, d, x, Nxy, Nz)
    return
end
function block_tridiagonal_solve!(a1::CuVector{T}, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, Nxy, Nz) where {T}
    function kernel(f, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, Nxy, Nz)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=Nxy
            f(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, (i-1) * Nz + 1, i * Nz)
        end
        return
    end
    @cuda threads=32 blocks=ceil(Int, Nxy/32) kernel(tridiagonal_solve!, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, Nxy, Nz)
    return
end
function tridiagonal_solve_nonblock!(P::SparseMatrixDIA, d, x, Nxy, Nz)
    block_tridiagonal_solve!(copy(P.diags[1].second), copy(P.diags[2].second), copy(P.diags[3].second), copy(d), x, Nxy, Nz)
    return x
end
function tridiagonal_solve_block!(P::SparseMatrixDIA, d, x, Nxy, Nz)
    block_tridiagonal_solve!(P.diags[2].second[1:2:end], P.diags[3].second[2:2:end], P.diags[1].second[1:2:end], P.diags[2].second[2:2:end],
                                  P.diags[4].second[1:2:end], P.diags[5].second[1:2:end], P.diags[3].second[1:2:end], P.diags[4].second[2:2:end],
                                  P.diags[6].second[1:2:end], P.diags[7].second[1:2:end], P.diags[5].second[2:2:end], P.diags[6].second[2:2:end], copy(d), x, Nxy, Nz)
    return x
end
