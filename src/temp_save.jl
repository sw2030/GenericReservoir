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


#####################
#### Block level tridiagonal solve
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


function inv_block_diag(W1, W2, W3, N)
    function kernel(W1, W2, W3, N)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=N
            invsmallw = inv(SMatrix{2,2}(W2[2i-1], W1[2i-1], W3[2i-1], W2[2i]))
            W2[2i-1] = invsmallw[1]
            W1[2i-1] = invsmallw[2]
            W3[2i-1] = invsmallw[3]
            W2[2i]   = invsmallw[4]
        end
    return
    end
    @cuda threads=256 blocks=ceil(Int, N/256) kernel(W1, W2, W3, N)
    return
end

function CPR_Setup!(J, P, E)
    #W = SparseMatrixDIA(Tuple(J.diags[i].first=>copy(J.diags[i].second) for i in [9, 10, 11]), size(J)...)
    #W.diags[1].second[2:2:end] .= zero(W.diags[1].second[2:2:end])
    #W.diags[3].second[2:2:end] .= zero(W.diags[3].second[2:2:end])
    #inv_block_diag(W.diags[1].second, W.diags[2].second, W.diags[3].second, 1122000);

    #Jidx = [2, 5, 8, 10, 12, 15, 18]
    #Wranges = [37401:2:2244000, 171:2:2244000, 3:2:2244000, 1:2:2244000, 1:2:2243998, 1:2:2243830, 1:2:2206600];
    #Jp = SparseMatrixDIA(Tuple((J.diags[Jidx[i]].first>>1)=>(W.diags[2].second[Wranges[i]] .* J.diags[Jidx[i]].second[1:2:end]) .+ (W.diags[3].second[Wranges[i]] .* J.diags[Jidx[i]-1].second[ceil(Int, i/4):2:end-floor(Int, i/5)]) for i in 1:7), size(J)[1]>>1, size(J)[1]>>1) # Create A_p
    Jidx = [2, 5, 8, 10, 12, 15, 18]
    Jp = SparseMatrixDIA(Tuple((J.diags[i].first>>1)=>J.diags[i].second[1:2:end] for i in Jidx), size(J,1)>>1, size(J,1)>>1)
    Pp = SparseMatrixDIA(Tuple(Jp.diags[i].first=>copy(Jp.diags[i].second) for i in [3,4,5]), size(Jp)...)
    Ep = SparseMatrixDIA(Tuple(Jp.diags[i].first=>copy(Jp.diags[i].second) for i in [1,2,6,7]), size(Jp)...)

    triLU!(P)
    triLU!(Pp)
    return Jp, Pp, Ep
end


 #=
            JAC, precP, precE = getjacobian(m, Δt, psgrid_new, psgrid_old)
            triLU!(precP, 1122000 ÷ 340, 340)

            print("LinSolve start...")
            gmresresult = gmres(JAC, RES, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->lsps_prec(precP, precE, n_prec, t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
            gmreserr = gmresresult[4]
            push!(errorlog, (steps, gmresnumcount, gmresresult[5]))
            println("...LinSolve done  ||  Iter : ", gmresresult[3], " || rel_err : ",gmreserr)
            if (iftol2 && gmreserr>arg2[1]) ## Option for having expensive solver
                print("(opt2) start...")
                gmresresult = gmres(JAC, RES, arg2[2];tol=tol_gmres, maxiter=arg2[3], M=(t->lsps_prec(precP, precE, arg2[4], t)), ifprint=false)
                gmresitercount += gmresresult[3]
                gmreserr = gmresresult[4]
                push!(errorlog, (steps, gmresnumcount, gmresresult[5]))
                println("...LinSolve done  ||  Iter : ", gmresresult[3], " || rel_err : ",gmreserr)
            end
            LinearAlgebra.axpy!(-1.0, gmresresult[1], psgrid_new) # Update step
            RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            @show norm_RES, norm_dg
            =#
