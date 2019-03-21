using DIA, LinearAlgebra

function create_P_E(J::SparseMatrixDIA)
    P = SparseMatrixDIA((0=>inv.(J.diags[4].second),), J.m, J.n)
    E = SparseMatrixDIA((J.diags[1], J.diags[2], J.diags[3], J.diags[5], J.diags[6], J.diags[7]), J.m, J.n)
    return P, E
end
function power_series_precond(P, E, x)
    result = copy(x)
    tmp = P*x
    LinearAlgebra.axpy!(-1.0, E*tmp, result) # result = x - EPx
    return P*result
end
    