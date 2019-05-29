using DIA, LinearAlgebra, CuArrays

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



   
