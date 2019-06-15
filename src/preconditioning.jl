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
function CPR_Setup!(J, P, E)
    Jidx = [2, 5, 8, 10, 12, 15, 18]
    Jp = SparseMatrixDIA(Tuple((J.diags[i].first>>1)=>J.diags[i].second[1:2:end] for i in Jidx), size(J,1)>>1, size(J,1)>>1)
    Pp = SparseMatrixDIA(Tuple(Jp.diags[i].first=>copy(Jp.diags[i].second) for i in [3,4,5]), size(Jp)...)
    Ep = SparseMatrixDIA(Tuple(Jp.diags[i].first=>Jp.diags[i].second for i in [1,2,6,7]), size(Jp)...)

    triLU!(P)
    triLU!(Pp)
    return Jp, Pp, Ep
end

function CPR_LSPS(J, P, E, Jp, Pp, Ep, RES, tol_gmres, n_iter, n_prec, n_restart, itercount)
    ## Page 3 of https://www.onepetro.org/download/journal-paper/SPE-106237-PA?id=journal-paper%2FSPE-106237-PA 
    xp = gmres(Jp, RES[1:2:end], n_restart;maxiter=n_iter, M=(t->lsps_prec(Pp, Ep, n_prec[1], t)), tol=tol_gmres) #(2)
    push!(itercount, xp[3])
    s = zero(RES)
    s[1:2:end] .+= xp[1] #(3)
    return s + lsps_prec(P, E, n_prec[2], RES-J*s)
end
