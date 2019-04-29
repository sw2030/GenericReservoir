using DIA, LinearAlgebra, CuArrays

function create_P_E(J::SparseMatrixDIA)
    P = SparseMatrixDIA((0=>map(inv, J.diags[10].second),), J.m, J.n)
    E = SparseMatrixDIA(Tuple(deleteat!([J.diags[i] for i in 1:length(J.diags)], 10)), J.m, J.n)
    return P, E
end
function power_series_precond(P, E, x)
    result = copy(x)
    BLAS.gemv!('N',-1.0, E, P*x, 1.0, result) # result = x - EPx
    return P*result
end
   
