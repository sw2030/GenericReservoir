## Here M is right preconditioner - for left preconditioner: look Reservoirnew
function gmres(A, b, restrt::Int64; tol::Real=1e-5, maxiter::Int=200, ifprint=false, M=identity, x_init = zero(b))
    x = copy(x_init)
    bnrm2, T =  norm(b), eltype(b)
    if bnrm2==zero(T) bnrm2 = one(T) end
    r = copy(b)
    BLAS.gemv!('N',-one(T), A, M(x), one(T), r)
    err = norm(r)/bnrm2
    itersave = 0
    ismax = false
    errlog = T[]

    restrt=min(restrt, length(b)-1)
    Q = [zero(b) for i in 1:restrt+1]
    H = zeros(T, restrt+1, restrt)
    cs = zeros(T, restrt)
    sn = zeros(T, restrt)
    s = zeros(T, restrt+1)
    flag = -1
    isave = 1
    y = zeros(restrt+1)
    for iter in 1:maxiter
        push!(errlog, err)
        itersave = iter
        r = Q[1]
        copyto!(r, b)
	BLAS.gemv!('N', -one(T), A, M(x), one(T), r)
	fill!(s, zero(T))
        s[1] = norm(r)
        rmul!(r, inv(s[1]))
        for i in 1:restrt
            isave = i
            w = Q[i+1]
	    BLAS.gemv!('N', one(T), A, M(Q[i]), zero(T), w)
            for k in 1:i
                H[k,i] = LinearAlgebra.dot(w, Q[k])
                LinearAlgebra.axpy!(-H[k,i],Q[k],w)
            end
            H[i+1,i] = norm(w)
            rmul!(w, inv(H[i+1,i]))
            for k in 1:i-1
                    temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
                    H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
                H[k,i]   = temp
            end

            cs[i], sn[i] = LinearAlgebra.givensAlgorithm(H[i, i], H[i+1, i])
            s[i+1] = -sn[i]*s[i]
            s[i]   = cs[i]*s[i]
            H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
	    H[i+1,i] = zero(T)
            err  = abs(s[i+1])/bnrm2

            if err < tol
                copyto!(y, s)
                ldiv!(UpperTriangular(view(H, 1:i, 1:i)), view(y, 1:i))
                for k in 1:i
                    LinearAlgebra.axpy!(y[k],Q[k],x)
                end
                flag = 0; break
            end
        end
        if  err < tol
            flag = 0
            break
        end
        copyto!(y, s)
        ldiv!(UpperTriangular(view(H, 1:restrt, 1:restrt)), view(y, 1:restrt))
        for k in 1:restrt
            LinearAlgebra.axpy!(y[k],Q[k],x)
        end
        copyto!(r, b)
	BLAS.gemv!('N', -one(T), A, M(x), one(T), r)
        s[isave+1] = norm(r)
        err = s[isave+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end
    end
    if flag==-1
        ifprint==true && print(" Maxiter")
        ismax = true
    end
    copyto!(x, M(x))
    return x, ismax, itersave, err, errlog
end
