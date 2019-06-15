using DIA, LinearAlgebra, CuArrays, StaticArrays

#### LSP preconditioner ### THREADS and Nxy/Nz size determined for SPE10 specifically
function lsps_prec(P, E::SparseMatrixDIA{T}, n, x) where {T}
    result = zero(x)
    triLU_solve!(P, x, result) #result = P^-1*x
    tmp1 = copy(result)
    tmp2 = zero(x)
    for i in 1:n
        BLAS.gemv!('N',  one(T), E, tmp1, zero(T), tmp2) # tmp2 = E * tmp1
	triLU_solve!(P, -tmp2, tmp1) # = tmp1 = -P^-1 * tmp2
        LinearAlgebra.axpy!(one(T), tmp1, result)
    end
    return result
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
    W = SparseMatrixDIA(Tuple(J.diags[i].first=>copy(J.diags[i].second) for i in [9, 10, 11]), size(J)...)
    W.diags[1].second[2:2:end] .= zero(W.diags[1].second[2:2:end])
    W.diags[3].second[2:2:end] .= zero(W.diags[3].second[2:2:end])
    inv_block_diag(W.diags[1].second, W.diags[2].second, W.diags[3].second, 1122000);

    Jidx = [2, 5, 8, 10, 12, 15, 18]
    Wranges = [37401:2:2244000, 171:2:2244000, 3:2:2244000, 1:2:2244000, 1:2:2243998, 1:2:2243830, 1:2:2206600];
    Jp = SparseMatrixDIA(Tuple((J.diags[Jidx[i]].first>>1)=>(W.diags[2].second[Wranges[i]] .* J.diags[Jidx[i]].second[1:2:end]) .+ (W.diags[3].second[Wranges[i]] .* J.diags[Jidx[i]-1].second[ceil(Int, i/4):2:end-floor(Int, i/5)]) for i in 1:7), size(J)[1]>>1, size(J)[1]>>1) # Create A_p
    Pp = SparseMatrixDIA(Tuple(Jp.diags[i].first=>copy(Jp.diags[i].second) for i in [3,4,5]), size(Jp)...)
    Ep = SparseMatrixDIA(Tuple(Jp.diags[i].first=>copy(Jp.diags[i].second) for i in [1,2,6,7]), size(Jp)...)

    triLU!(P)
    triLU!(Pp)
    return Jp, Pp, Ep, W
end

function CPR_LSPS(J, P, E, Jp, Pp, Ep, W, RES, tol_gmres, n_iter, n_prec, n_restart, itercount)
    ## Page 3 of https://www.onepetro.org/download/journal-paper/SPE-106237-PA?id=journal-paper%2FSPE-106237-PA 
    xp = gmres(Jp, (W*RES)[1:2:end], n_restart;maxiter=n_iter, M=(t->lsps_prec(Pp, Ep, n_prec[1], t)), tol=tol_gmres) #(2)
    push!(itercount, xp[3])
    s = zero(RES)
    s[1:2:end] .+= xp[1] #(3)
    return s + lsps_prec(P, E, n_prec[2], RES-J*s)
end
