using LinearAlgebra

function Solve_adaptive(m::Reservoir_Model, t_init, Δt, g_guess, n_steps; tol_relnorm=1e-3, tol_gmres=5e-3, n_restart=20, n_iter=200, precondf=GenericReservoir.power_series_precond)
     
    ## Initialize
    record_p   = zeros(2, n_steps)
    psgrid_old = copy(g_guess)
    psgrid_new = copy(psgrid_old)
    
    ## Time stepping start
    for steps in 1:n_steps
        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        norm_RES_save = norm(RES)
        norm_RES = norm_RES_save
        println("\nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
        gmresnumcount, gmresitercount, norm_dg = 0, 0, 1.0
        while(norm_RES/norm_RES_save > tol_relnorm)
            
            ## In case it is diverging
            if (norm_RES > 5.0e6 || gmresnumcount > 9 || (gmresnumcount>4 && norm_RES>1.0e4) || norm_dg < 1e-2)
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
                gmresnumcount, gmresitercount = 0, 0
                RES = getresidual(m, Δt, psgrid_new, psgrid_old)
		norm_RES = norm(RES)
		println("\nDiverged, Δt adapted... Δt : ",Δt*2.0, "->", Δt,"\n\nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
            end
            
            JAC, precP, precE = getjacobian(m, Δt, psgrid_new, psgrid_old)
                       
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
	record_p[:, steps] = [t_init+Δt; 2.0*sum(psgrid_old[1:2:end])/length(psgrid_old)]
        println("Total GMRES iteration : ",gmresitercount, " | Avg p : ", record_p[2, steps]," | Total time : ", t_init+Δt, " Days")
        t_init += Δt
	if Δt<50 Δt *= 2.0 end
    end
    print("\nSolve done")
    return psgrid_new, record_p
end
