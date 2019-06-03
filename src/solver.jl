using LinearAlgebra

function Solve_adaptive(m::Reservoir_Model, t_init, Δt, g_guess, n_steps; tol_relnorm=1e-3, tol_gmres=1e-2, n_restart=20, n_iter=50, n_prec=7, step_init=0, iftol2=false, arg2=(1e-1, 100, 30, 31))
     
    ## Initialize
    record_p   = zeros(2, n_steps)
    psgrid_old = copy(g_guess)
    psgrid_new = copy(psgrid_old)
    errorlog   = []
    println("Preconditioner degree : ", n_prec)
    println("GMRES tol : ", tol_gmres)
    print("Option for More expensive Linsol if GMRES diverges : ")
    if iftol2 println("On \nIf err > ", arg2[1], " | Restart : ", arg2[2], " | Maxiter : ", arg2[3], "| Prec degree : ", arg2[4]) 
    else println("Off")
    end
    runtime = 0.0
    itercount_total_c, itercount_total_d, t_init_save = 0, 0, t_init
    ## Time stepping start
    for steps in 1:n_steps
	stepruntime = @elapsed begin
        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        norm_RES_save = norm(RES)
        norm_RES = norm_RES_save
        println("\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
        gmresnumcount, gmresitercount, norm_dg, gmreserr, gmresresult, itercount_div = 0, 0, 1.0, 0.001, 0.0, 0
        while(norm_RES/norm_RES_save>tol_relnorm)
            
            ## In case it is diverging
	    if (norm_RES > 5.0e6 || gmresnumcount > 9 || (gmresnumcount>6 && norm_RES>1.0e4) || (gmreserr > 0.9 && norm_RES < 50) || gmreserr > 0.99) # These are the conditions for reducing Δt (Not converged)
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
		itercount_div += gmresitercount
                gmresnumcount, gmresitercount = 0, 0
                RES = getresidual(m, Δt, psgrid_new, psgrid_old)
		norm_RES = norm(RES)
		println("\nNot Converged, Δt reduced... Δt : ",Δt*2.0, "->", Δt,"\n\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
            end
            
            JAC, precP, precE = getjacobian(m, Δt, psgrid_new, psgrid_old)
                       
            print("GMRES start...")
	    gmresresult = gmres(JAC, RES, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->lsps_prec(precP, precE, n_prec, t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
	    gmreserr = gmresresult[4]
	    push!(errorlog, (steps, gmresnumcount, gmresresult[5]))
	    println("...GMRES done  ||  Iter : ", gmresresult[3], " || rel_err : ",gmreserr)
	    if (iftol2 && gmreserr>arg2[1]) ## Option for having expensive solver
		print("(opt2)start...")
	        gmresresult = gmres(JAC, RES, arg2[2]; tol=tol_gmres, maxiter=arg2[3], M=(t->lsps_prec(precP, precE, arg2[4], t)), ifprint=false)
		gmresitercount += gmresresult[3]
                gmreserr = gmresresult[4]
         	push!(errorlog, (steps, gmresnumcount, gmresresult[5]))
                println("...GMRES done  ||  Iter : ", gmresresult[3], " || rel_err : ",gmreserr)
	    end
            LinearAlgebra.axpy!(-1.0, gmresresult[1], psgrid_new) # Update step
	    RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            @show norm_RES, norm_dg        
	    if isnan(norm_RES) norm_RES=1.0e10 end
        end
        copyto!(psgrid_old, psgrid_new)
	record_p[:, steps] = [t_init+Δt; 2.0*sum(psgrid_old[1:2:end])/length(psgrid_old)]
	println("GMRES iteration(converge) : ",gmresitercount, " | GMRES iteration(diverge) : ", itercount_div)
	println("Min p : ", minimum(psgrid_old[1:2:end]), " | Max p : ", maximum(psgrid_old[1:2:end]))
	println("Min s : ", minimum(psgrid_old[2:2:end]), " | Max s : ", maximum(psgrid_old[2:2:end]))
	println("Avg p : ", record_p[2, steps], " | Total time : ", t_init+Δt, " Days")
	t_init += Δt
	itercount_total_c += gmresitercount
	itercount_total_d += itercount_div
	if t_init>2000 break end
	if (((gmresitercount < 5*n_iter && gmresnumcount < 7) || gmresitercount < 100))
	    if (Δt>25.0&& Δt<=50.0) Δt=50.0 end
	    if Δt<=25.0 Δt *= 2.0 end
        end
        end
	runtime += stepruntime
	println("Runtime on this step : ", stepruntime, " | Total Runtime : ", runtime) 
    end
    println("\nTotal Days : ",t_init-t_init_save ," | Total iteration(converge) : ", itercount_total_c, " | Total iteration(diverge) : ", itercount_total_d)
    return psgrid_new, record_p, errorlog
end
