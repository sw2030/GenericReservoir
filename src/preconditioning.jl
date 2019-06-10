using DIA, LinearAlgebra, CuArrays, StaticArrays

function tridiagonal_solve!(a::AbstractVector{T}, b::AbstractVector{T}, c::AbstractVector{T}, d::AbstractVector{T}, x::AbstractVector{T}, startidx, endidx) where {T} # b, d, x from startidx:endidx, a, c from startidx:endidx-1
    for i in startidx+1:endidx
	b[i] = b[i] - a[i-1]*c[i-1]/b[i-1]
	d[i] = d[i] - a[i-1]*d[i-1]/b[i-1]
    end
    x[endidx] = d[endidx]/b[endidx]
    for i in (endidx-1):-1:startidx
	x[i] = (d[i] - c[i]*x[i+1])/b[i]
    end
end
function block_tridiagonal_solve!(a::CuVector{T}, b::CuVector{T}, c::CuVector{T}, d::CuVector{T}, x::CuVector{T}, Nx, Ny, Nz) where {T} ## Assume Input is Nx*Ny*Nz length vectors(b, d, x)
    function kernel(f, a, b, c, d, x, Nx, Ny, Nz)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y

	if i<=Nx && j<=Ny
            startid = Nz*((i-1)*Ny + j-1) + 1
            endid = startid + Nz - 1
	    f(a, b, c, d, x, startid, endid)
	end
	return 
    end

    max_threads = 256
    threads_x   = min(max_threads, Nx)
    threads_y   = min(max_threads ÷ threads_x, Ny)
    threads     = (threads_x, threads_y)
    blocks      = ceil.(Int, (Nx, Ny) ./ threads)

    @cuda threads=threads blocks=blocks kernel(tridiagonal_solve!, a, b, c, d, x, Nx, Ny, Nz)
    return
end

function tridiagonal_block_solve!(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, startidx, endidx) # a, b, c 4 Vectors each, d, x are gonna be long vector
    for i in startidx+1:endidx
	A = SMatrix{2,2}(a1[i-1], a3[i-1], a2[i-1], a4[i-1])
	Binv = inv(SMatrix{2,2}(b1[i-1], b3[i-1], b2[i-1], b4[i-1]))
	Bdiff = A*Binv*SMatrix{2,2}(c1[i-1], c3[i-1], c2[i-1], c4[i-1])
	Ddiff = A*Binv*SVector{2}(d[2i-3], d[2i-2])
	b1[i]  -= Bdiff[1]
	b2[i]  -= Bdiff[3]
	b3[i]  -= Bdiff[2]
	b4[i]  -= Bdiff[4]
	d[2i-1] -= Ddiff[1]
	d[2i]   -= Ddiff[2]
    end
    xendidx = inv(SMatrix{2,2}(b1[endidx], b3[endidx], b2[endidx], b4[endidx]))*SVector{2}(d[2endidx-1], d[2endidx])
    x[2endidx-1] = xendidx[1]
    x[2endidx] = xendidx[2]
    for i in (endidx-1):-1:startidx
        xi = inv(SMatrix{2,2}(b1[i], b3[i], b2[i], b4[i]))*(SVector{2}(d[2i-1], d[2i]) - SMatrix{2,2}(c1[i], c3[i], c2[i], c4[i]) * SVector{2}(x[2i+1], x[2i+2]))
	x[2i-1] = xi[1]
	x[2i] = xi[2]
    end
end
function block_tridiagonal_block_solve!(a1::CuVector{T}, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, Nx, Ny, Nz) where {T}
    function kernel(f, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, Nx, Ny, Nz)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y

	if i<=Nx && j<=Ny
	    startid = Nz*((i-1)*Ny + j-1) + 1
            endid = startid + Nz - 1
	    f(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, startid, endid)
	end
	return
    end

    max_threads = 256
    threads_x   = min(max_threads, Nx)
    threads_y   = min(max_threads ÷ threads_x, Ny)
    threads     = (threads_x, threads_y)
    blocks      = ceil.(Int, (Nx, Ny) ./ threads)

    @cuda threads=threads blocks=blocks kernel(tridiagonal_block_solve!, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d, x, Nx, Ny, Nz)
    return 
end

function lsps_prec(P, E::SparseMatrixDIA{T}, n, x) where {T}
    result = zero(x)
    block_tridiagonal_block_solve!(P.diags[2].second[1:2:end], P.diags[3].second[2:2:end], P.diags[1].second[1:2:end], P.diags[2].second[2:2:end],
                                  P.diags[4].second[1:2:end], P.diags[5].second[1:2:end], P.diags[3].second[1:2:end], P.diags[4].second[2:2:end],
				  P.diags[6].second[1:2:end], P.diags[7].second[1:2:end], P.diags[5].second[2:2:end], P.diags[6].second[2:2:end], copy(x), result, 60, 220, 85) #result = P^-1*x
    tmp1 = copy(result)
    tmp2 = zero(x)
    for i in 1:n
	BLAS.gemv!('N',  one(T), E, tmp1, zero(T), tmp2) # tmp2 = E * tmp1
	block_tridiagonal_block_solve!(P.diags[2].second[1:2:end], P.diags[3].second[2:2:end], P.diags[1].second[1:2:end], P.diags[2].second[2:2:end],
				  P.diags[4].second[1:2:end], P.diags[5].second[1:2:end], P.diags[3].second[1:2:end], P.diags[4].second[2:2:end], 
				  P.diags[6].second[1:2:end], P.diags[7].second[1:2:end], P.diags[5].second[2:2:end], P.diags[6].second[2:2:end], -tmp2, tmp1, 60, 220, 85)
	#BLAS.gemv!('N', -one(T), P, tmp2, zero(T), tmp1) # = tmp1 = -P^-1 * tmp2 
	LinearAlgebra.axpy!(one(T), tmp1, result)
    end
    return result
end
function lsps_prec_CPR(P, E::SparseMatrixDIA{T}, n, x) where {T}
    result = zero(x)
    block_tridiagonal_solve!(copy(P.diags[1].second), copy(P.diags[2].second), copy(P.diags[3].second), copy(x), result, 60, 220, 85) 				  #result = P^-1*x
    tmp1 = copy(result)
    tmp2 = zero(x)
    for i in 1:n
        BLAS.gemv!('N',  one(T), E, tmp1, zero(T), tmp2) # tmp2 = E * tmp1
	block_tridiagonal_solve!(copy(P.diags[1].second), copy(P.diags[2].second), copy(P.diags[3].second), -tmp2, tmp1, 60, 220, 85)
	#BLAS.gemv!('N', -one(T), P, tmp2, zero(T), tmp1) # = tmp1 = -P^-1 * tmp2
        LinearAlgebra.axpy!(one(T), tmp1, result)
    end
    return result
end
function inv_block_diag(W1, W2, W3, N)
    function kernel(W1, W2, W3, N)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
	if i<=N
	    smallw = SMatrix{2,2}(W2[2i-1], W1[2i-1], W3[2i-1], W2[2i])
	    invsmallw = inv(smallw)
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
	
    
function CPR_LSPS(J, P, E, RES, tol_gmres, n_iter, n_prec, n_restart)
    ## Page 3 of https://www.onepetro.org/download/journal-paper/SPE-106237-PA?id=journal-paper%2FSPE-106237-PA 
    W = SparseMatrixDIA(Tuple(J.diags[i].first=>copy(J.diags[i].second) for i in [9, 10, 11]), size(J)...)
    W.diags[1].second[2:2:end] .= zero(W.diags[1].second[2:2:end])
    W.diags[3].second[2:2:end] .= zero(W.diags[3].second[2:2:end])
    inv_block_diag(W.diags[1].second, W.diags[2].second, W.diags[3].second, 1122000);
    rp = (W*RES)[1:2:end] #(1)
    Jidx = [2, 5, 8, 10, 12, 15, 18]
    Wranges = [37401:2:2244000, 171:2:2244000, 3:2:2244000, 1:2:2244000, 1:2:2243998, 1:2:2243830, 1:2:2206600];
    Jp = SparseMatrixDIA(Tuple((J.diags[Jidx[i]].first>>1)=>(W.diags[2].second[Wranges[i]] .* J.diags[Jidx[i]].second[1:2:end]) .+ (W.diags[3].second[Wranges[i]] .* J.diags[Jidx[i]-1].second[ceil(Int, i/4):2:end-floor(Int, i/5)]) for i in 1:7), length(RES)>>1, length(RES)>>1) # Create A_p
    Pp = SparseMatrixDIA(Tuple(Jp.diags[i].first=>Jp.diags[i].second for i in [3,4,5]), size(Jp)...)
    Ep = SparseMatrixDIA(Tuple(Jp.diags[i].first=>Jp.diags[i].second for i in [1,2,6,7]), size(Jp)...)
    xp = gmres(Jp, rp, n_restart[1];maxiter=n_iter[1], M=(t->lsps_prec_CPR(Pp, Ep, n_prec[1], t)), tol=tol_gmres[1]) #(2)
    s = zero(RES)
    s[1:2:end] .+= xp[1] #(3)
    x = gmres(J, RES-J*s, n_restart[2];maxiter=n_iter[2], M=(t->lsps_prec(P, E, n_prec[2], t)), tol=tol_gmres[2]) #(4), (5)
    x[1] .+= s #(6)
    return x, xp[3]
end


   
