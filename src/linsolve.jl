function LinSolve_LSPS(m, Δt, RES, psgrid_old, psgrid_new, args, n_prec;linsolve=fgmres)
    jac_time = @elapsed CuArrays.@sync JAC, precP, precE, diagW = getjacobian_scaled(m, Δt, psgrid_new, psgrid_old)
    copyto!(RES, diagW*RES)
    triLU!(precP)

    linsol_time = @elapsed CuArrays.@sync begin
        print("LinSolve start...")
        gmresresult = linsolve(JAC, RES, args[1];maxiter=args[2], M=(t->lsps_prec(precP, precE, n_prec, t)), tol=args[3]);
        println("...LinSolve done  ||  Iter : ", gmresresult[2], " || rel_err : ", gmresresult[3])
    end
    
    return gmresresult[1], jac_time, linsol_time, (gmresresult[2], 0) ## 0 because no inner iteration
end

function LinSolve_CPR_LSPS(m, Δt, RES, psgrid_old, psgrid_new, args, CPR_args;linsolve=fgmres)
    jac_time = @elapsed CuArrays.@sync JAC, precP, precE, diagW = getjacobian_scaled(m, Δt, psgrid_new, psgrid_old)
    copyto!(RES, diagW*RES)
    Jp, Pp, Ep = CPR_Setup!(JAC, precP, precE)
    inneriter=[]
    linsol_time = @elapsed CuArrays.@sync begin
        print("LinSolve start...")
        gmresresult = linsolve(JAC, RES, args[1];maxiter=args[2], 
                                M=(t->CPR_LSPS(JAC, precP, precE, Jp, Pp, Ep, t, CPR_args, inneriter)), tol=args[3]);
        println("...LinSolve done  ||  Iter : ", (gmresresult[2], sum(inneriter)), " || rel_err : ", gmresresult[3])
    end
    
    return gmresresult[1], jac_time, linsol_time, (gmresresult[2], sum(inneriter))
end

function LinSolve_CPR_MG(m, Δt, RES, psgrid_old, psgrid_new, args, CPR_args;linsolve=fgmres)
    jac_time = @elapsed CuArrays.@sync JAC, precP, precE = getjacobian_frs(m, Δt, psgrid_new, psgrid_old)
    RES[1:2:end] .+= RES[2:2:end]    
    Jp, Pp, Ep = CPR_Setup!(JAC, precP, precE)
    inneriter=[]
    ml = DIA.gmg(Jp, (85, 220, 60), (1,2,2))
    linsol_time = @elapsed CuArrays.@sync begin
        print("LinSolve start...")
        gmresresult = linsolve(JAC, RES, args[1];maxiter=args[2],
                                M=(t->CPR_MG(JAC, precP, precE, Jp, ml, t, CPR_args, inneriter)), tol=args[3]);
        println("...LinSolve done  ||  Iter : ", (gmresresult[2], sum(inneriter)), " || rel_err : ", gmresresult[3])
    end

    return gmresresult[1], jac_time, linsol_time, (gmresresult[2], sum(inneriter))
end
