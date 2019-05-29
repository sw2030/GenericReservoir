using DIA, LinearAlgebra, CuArrays

function create_P_E(J::SparseMatrixDIA)
    P = SparseMatrixDIA((0=>map(inv, J.diags[10].second),), J.m, J.n)
    E = SparseMatrixDIA(Tuple(deleteat!([J.diags[i] for i in 1:length(J.diags)], 10)), J.m, J.n)
    return P, E
end
function ps_precond_ord1(P, E, x)
    result = copy(x)
    BLAS.gemv!('N',-1.0, E, P*x, 1.0, result) # result = x - EPx
    return P*result
end
function ps_precond_ord2(P, E, x)
    tmp1 = copy(x)
    tmp2 = P*x
    BLAS.gemv!('N', -1.0, E, tmp2, 1.0, tmp1)   # tmp1 = x - EPx
    BLAS.gemv!('N', 1.0, P, tmp1, 0.0, tmp2) # tmp2 = P(x-EPx)
    copyto!(tmp1, x)              # tmp1 = x
    BLAS.gemv!('N', -1.0, E, tmp2, 1.0, tmp1)   # tmp1 = x - E(P(x-EPx))
    BLAS.gemv!('N', 1.0, P, tmp1, 0.0, tmp2) # tmp2 = P(x-E(P(x-EPx)))
    return tmp2
end

function lsps_prec(P, E, n, x)
    result = P*x
    tmp1 = copy(result)
    tmp2 = zero(x)
    for i in 1:n
        BLAS.gemv!('N',  1.0, E, tmp1, 0.0, tmp2) # tmp2 = E * tmp1
	BLAS.gemv!('N', -1.0, P, tmp2, 0.0, tmp1) # = tmp = -P * tmp2 
	LinearAlgebra.axpy!(1.0, tmp1, result)
    end
    return result
end



   
