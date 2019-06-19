using LinearAlgebra

function Solve_SPE10(m::Reservoir_Model{T}, t_init, Δt, g_guess, n_steps;prec="CPR-LSPS", tol_relnorm=1e-3, tol_gmres=1e-2, n_restart=10, n_iter=10, n_prec=7, step_init=0, iftol2=false, arg2=(1e-1, 100, 30, 31), CPR_tol=1e-2, CPR_iter=10, CPR_prec=(7, 7), CPR_restart=10) where {T}


	record_p   = zeros(2, n_steps)
    	psgrid_old = copy(g_guess)
    	psgrid_new = copy(psgrid_old)
    	errorlog   = []
    	runtime = 0.0
    	itercount_total_c, itercount_total_d, t_init_save = 0, 0, t_init
	#### Print Arguments
	print_init(tol_relnorm, tol_gmres, n_restart, n_iter, CPR_prec, CPR_restart, CPR_iter, CPR_tol, arg2, prec, n_prec, iftol2)

	#### Time stepping start
    	for steps in 1:n_steps
        	stepruntime = @elapsed begin
		
		#### Initialize Each Steps
	        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        	norm_RES_save = norm(RES)
        	norm_RES = norm_RES_save
        	println("\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
        	gmresnumcount, gmresitercount, itercount_div, inneritercount = 0, 0, 0, 0

        	while(norm_RES/norm_RES_save>tol_relnorm)
            	#### In case it is Divergine
	    		if ((norm_RES>1e5 && steps>1) || gmresnumcount > 9 || (gmresnumcount>6 && norm_RES>1.0e4)) ### These are the conditions for reducing Δt (Not converged)
                		copyto!(psgrid_new, psgrid_old)
                		Δt *= 0.5
                		itercount_div += gmresitercount
                		gmresnumcount, gmresitercount = 0, 0
                		RES = getresidual(m, Δt, psgrid_new, psgrid_old)
                		norm_RES = norm(RES)
                		println("\nNot Converged, Δt reduced... Δt : ",Δt*2.0, "->", Δt,"\n\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
			end
	    		#### Calculate JAC, RES
	    		JAC, precP, precE, diagW = getjacobian2(m, Δt, psgrid_new, psgrid_old)
			RES_scaled = diagW * RES
			
			#### Linsolve
			print("LinSolve start...")
			dg, gmreserr, Linsolveiter, log = prec=="CPR-LSPS" ? CPR_LinSolve(RES_scaled, JAC, precP, precE, CPR_tol, CPR_iter, CPR_prec, CPR_restart, tol_gmres, n_iter, n_restart) : LSPS_LinSolve(RES, J, P, E, tol_gmres, n_prec, n_iter, n_restart)
			println("...LinSolve done  ||  Iter : ",Linsolveiter, " || rel_err : ",gmreserr)
	
		   	#### Update and Print
			push!(errorlog, log)
			gmresnumcount += 1
			gmresitercount += Linsolveiter[1]
			prec=="CPR-LSPS" ? inneritercount+=Linsolveiter[2] : nothing
			LinearAlgebra.axpy!(-1.0, dg, psgrid_new) # Update step
			RES = getresidual(m, Δt, psgrid_new, psgrid_old)
			norm_RES, norm_dg = norm(RES), norm(dg)
			@show norm_RES, norm_dg
			if isnan(norm_RES) norm_RES=1.0e10 end
		end
		griddiff = psgrid_new - psgrid_old
	        copyto!(psgrid_old, psgrid_new)
       		record_p[:, steps] = [t_init+Δt; 2*sum(psgrid_old[1:2:end])/length(psgrid_old)]
		print_final(steps, t_init, Δt, psgrid_old, record_p, gmresitercount, itercount_div, inneritercount, griddiff)
		t_init += Δt
		itercount_total_c += gmresitercount
		itercount_total_d += itercount_div
		if t_init>2000 break end
		if (gmresnumcount < 8)
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

function print_final(steps, t_init, Δt, psgrid_old, record_p, gmresitercount, itercount_div, inneritercount, griddiff)
	println("CPR Stage 1 iteration : ", inneritercount, "| max dp, ds : ", (maximum(abs.(Array(griddiff[1:2:end]))), maximum(abs.(Array(griddiff[2:2:end])))))
        println("GMRES iteration(converge) : ",gmresitercount, " | GMRES iteration(diverge) : ", itercount_div)
        println("Min p : ", minimum(psgrid_old[1:2:end]), " | Max p : ", maximum(psgrid_old[1:2:end]))
        println("Min s : ", minimum(psgrid_old[2:2:end]), " | Max s : ", maximum(psgrid_old[2:2:end]))
        println("Avg p : ", record_p[2, steps], " | Total time : ", t_init+Δt, " Days")
end

function print_init(tol_relnorm, tol_gmres, n_restart, n_iter, CPR_prec, CPR_restart, CPR_iter, CPR_tol, arg2, prec, n_prec, iftol2)
        println("CONVERGE_RELNORM : ", tol_relnorm, " | GMRES tol : ", tol_gmres," | GMRES restart : ", n_restart, " | GMRES maxiter : ", n_iter)
        if prec=="CPR-LSPS" println("Preconditioning : CPR-LSPS \n", "CPR-LSPS degree : ", CPR_prec, "| CPR restart : ", CPR_restart, " | CPR maxiter : ", CPR_iter, " | CPR gmres tol : ",CPR_tol) end
        if prec=="LSPS" println("Preconditioning : LSPS\n", "LSPS degree : ", n_prec) end
        print("Option for More expensive Linsol if GMRES diverges : ")
        if iftol2 println("On \nIf err > ", arg2[1], " | Restart : ", arg2[2], " | Maxiter : ", arg2[3], "| Prec degree : ", arg2[4])
        else println("Off")
        end
end

function CPR_LinSolve(RES, J, P, E, CPR_tol, CPR_iter, CPR_prec, CPR_restart, tol_gmres, n_iter, n_restart)
    	Jp, Pp, Ep = CPR_Setup!(J, P, E)
    	inneriter = []
    	gmresresult = gmres(J, RES, n_restart;maxiter=n_iter, M=(t->CPR_LSPS(J, P, E, Jp, Pp, Ep, t, CPR_tol, CPR_iter, CPR_prec, CPR_restart, inneriter)), tol=tol_gmres)
	return gmresresult[1], gmresresult[4], (gmresresult[3], sum(inneriter)), gmresresult[5]
end
function LSPS_LinSolve(RES, J, P, E, tol_gmres, n_prec, n_iter, n_restart)
	gmresresult = gmres(J, RES, n_restart;maxiter=n_iter, M=(t->lsps_prec(P, E, n_prec, t)), tol=tol_gmres);
	return gmresresult[1], gmresresult[4], gmresresult[3], gmresresult[5]
end
