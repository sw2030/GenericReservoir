function Solve_adaptive(m::Reservoir_Model, t_init, Δt, g_guess, n_steps; tol_relnorm=1e-2, tol_gmres=1e-2, n_restart=20, n_iter=1000, precondf=GenericReservoir.precond_1)
     
    ## Initialize
    record_p   = zeros(2, n_steps)
    psgrid_old = copy(g_guess)
    psgrid_new = copy(psgrid_old)
    
    ## Time stepping start
    for steps in 1:n_steps
        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        norm_RES_save = norm(RES)
        norm_RES = norm_RES_save
        println("\nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt, " | ΣΔt : ", t_init+Δt)
        gmresnumcount, gmresitercount = 0, 0
        while(norm_RES/norm_RES_save > tol_relnorm)
            
            ## In case it is diverging
            if (norm_RES > 1.0e6 || gmresnumcount > 9 || (gmresnumcount>4 && norm_RES>1.0e4))
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
                println("\nNew Δt adapted... | Δt : ",Δt, " | ΣΔt : ", t_init+Δt)
                gmresnumcount, gmresitercount = 0, 0
                if backend=='D' close(RES) end
                RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            end
            
            JAC = getjacobian(m, Δt, psgrid_new, psgrid_old)
            
            
            JAC_GPU = DtoCu(JAC)
            RES_GPU = DtoCu(RES)
            precP, precE = Reservoir.make_P_E_precond_1(JAC_GPU)
            print("GMRES start...")
            gmresresult = stencilgmres2(JAC_GPU, RES_GPU, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->precondf(precP,precE,t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
            if backend=='D' close(RES), close(JAC) end
            println("...GMRES done  ||  Iter : ", gmresresult[3])
            gmres_dist = CutoD(gmresresult[1])
            LinearAlgebra.axpy!(-1.0, gmres_dist, psgrid_new)
            RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            if backend=='D' close(gmres_dist) end
            @show norm_RES, norm_dg        
        end
        if backend=='D' close(RES) end
        copyto!(psgrid_old, psgrid_new)
        record_p[:, steps] = [t_init+Δt; sum(Array(psgrid_old[1].A))/60/220/85]
        println("Total GMRES iteration : ",gmresitercount, " | Avg p : ", record_p[2, steps])
        t_init += Δt
        Δt *= 2.0
    end
    if backend=='D' close(psgrid_old) end
    print("\nSolve done")
    return psgrid_new, record_p
end