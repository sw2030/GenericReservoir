using LinearAlgebra

function Solve_adaptive(m::Reservoir_Model, t_init, Δt, g_guess, n_steps; tol_relnorm=1e-3, tol_gmres=1e-2, n_restart=20, n_iter=1000, precondf=GenericReservoir.power_series_precond)
     
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
            if (norm_RES > 5.0e6 || gmresnumcount > 9 || (gmresnumcount>4 && norm_RES>1.0e4))
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
                println("\nNew Δt adapted... \nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt, " | ΣΔt : ", t_init+Δt)
                gmresnumcount, gmresitercount = 0, 0
                RES = getjacobian(m, Δt, psgrid_new, psgrid_old)
            end
            
            JAC = getjacobian(m, Δt, psgrid_new, psgrid_old)
            precP, precE = create_P_E(JAC)
            
            print("GMRES start...")
            gmresresult = gmres(JAC, RES, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->precondf(precP,precE,t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
            println("...GMRES done  ||  Iter : ", gmresresult[3])
            
            LinearAlgebra.axpy!(-1.0, gmresresult[1], psgrid_new) # Update step
            RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            @show norm_RES, norm_dg        
        end
        copyto!(psgrid_old, psgrid_new)
        record_p[:, steps] = [t_init+Δt; sum(psgrid_old)[1]/length(psgrid_old)]
        println("Total GMRES iteration : ",gmresitercount, " | Avg p : ", record_p[2, steps])
        t_init += Δt
        Δt *= 2.0
    end
    print("\nSolve done")
    return psgrid_new, record_p
end