using DIA, LinearAlgebra, CuArrays, StaticArrays

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
        block_execute!(tridiagonal_LU_solve_block!, 32, 1122000, 170, P.diags[1].second, P.diags[2].second, P.diags[3].second, P.diags[4].second, P.diags[5].second, P.diags[6].second, P.diags[7].second, copy(d), x)
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
function tridiagonal_LU_solve!(a::AbstractVector{T}, b::AbstractVector{T}, c::AbstractVector{T}, d::AbstractVector{T}, x::AbstractVector{T}, startidx, endidx) where {T}
    @inbounds for i in startidx+1:endidx
        d[i] -= a[i-1]*d[i-1]
    end
    x[endidx] = d[endidx]/b[endidx]
    @inbounds for i in (endidx-1):-1:startidx
        x[i] = (d[i] - c[i]*x[i+1])/b[i]
    end
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
	b2[2i-1] -= LCT[1]
	b3[2i-1] -= LCT[2]
	b1[2i-1] -= LCT[3]
	b2[2i]   -= LCT[4]
    end
end
function tridiagonal_LU_solve_block!(a1, a2, b1, b2, b3, c1, c2, d, x, startidx, endidx)
    @inbounds for i in startidx+1:endidx
        AD = SMatrix{2,2}(a2[2i-3], a1[2i-3], b1[2i-2], a2[2i-2])*SVector{2}(d[2i-3], d[2i-2])
        d[2i-1] -= AD[1]
        d[2i]   -= AD[2]
    end
    xendidx = SMatrix{2,2}(b2[2endidx-1], b1[2endidx-1], b3[2endidx-1], b2[2endidx])\SVector{2}(d[2endidx-1], d[2endidx])
    x[2endidx-1] = xendidx[1]
    x[2endidx]   = xendidx[2]
    @inbounds for i in (endidx-1):-1:startidx
        binvdminuscx = SMatrix{2,2}(b2[2i-1], b1[2i-1], b3[2i-1], b2[2i])\(SVector{2}(d[2i-1], d[2i]) - SMatrix{2,2}(c1[2i-1], b3[2i], c2[2i-1], c1[2i]) * SVector{2}(x[2i+1], x[2i+2]))
        x[2i-1] = binvdminuscx[1]
        x[2i]   = binvdminuscx[2]
    end
end

