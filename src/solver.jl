using LinearAlgebra

function Solve_SPE10(m::Reservoir_Model{T}, t_init, Δt, g_guess, n_steps;prec="CPR-LSPS", tol_relnorm=1e-3, tol_gmres=1e-2, n_restart=10, n_iter=10, n_prec=7, step_init=0, CPR_tol=1e-1, CPR_iter=10, CPR_prec=(7, 7), CPR_restart=10, iternumtol=7, linsolf=fgmres) where {T}


	record_p   = zeros(2, n_steps)
    	psgrid_old = copy(g_guess)
    	psgrid_new = copy(psgrid_old)
    	runtime = 0.0
	prod_rec = []
    	itercount_total_c, itercount_total_d, t_init_save, jac_time_total, linsol_time_total= 0, 0, t_init, 0.0, 0.0
	#### Print Arguments
	print_init(tol_relnorm, tol_gmres, n_restart, n_iter, CPR_prec, CPR_restart, CPR_iter, CPR_tol, prec, n_prec)
	
	#### Time stepping start
    	for steps in 1:n_steps
        	stepruntime = @elapsed begin
		
		#### Initialize Each Steps
	        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        	norm_RES_save = norm(RES)
        	norm_RES = norm_RES_save
        	println("\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
        	gmresnumcount, gmresitercount, itercount_div, inneritercount, jac_time, linsol_time = 0, 0, 0, 0, 0.0, 0.0

        	while(norm_RES/norm_RES_save>tol_relnorm)
            	#### In case it is Divergine
	    		if ((norm_RES>1e5 && steps>1) || gmresnumcount > 9 || (gmresnumcount>6 && norm_RES>1.0e4)) ### These are the conditions for reducing Δt (Not converged)
                		copyto!(psgrid_new, psgrid_old)
                		Δt *= 0.5
                		itercount_div += gmresitercount
                		gmresnumcount, gmresitercount = 0, 0
                		getresidual!(m, Δt, psgrid_new, psgrid_old, RES)
                		norm_RES = norm(RES)
                		println("\nNot Converged, Δt reduced... Δt : ",Δt*2.0, "->", Δt,"\n\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
			end
	    		#### Calculate JAC, RES
			jac_time = @elapsed JAC, precP, precE, diagW = getjacobian_scaled(m, Δt, psgrid_new, psgrid_old)
			copyto!(RES, diagW*RES)
			#JAC, precP, precE = getjacobian(m, Δt, psgrid_new, psgrid_old)
			
			#### Linsolve
			_linsol_time = @elapsed begin
			print("LinSolve start...")
			dg, gmreserr, Linsolveiter, log = prec=="CPR-LSPS" ? CPR_LinSolve(RES, JAC, precP, precE, CPR_tol, CPR_iter, CPR_prec, CPR_restart, tol_gmres, n_iter, n_restart;f=linsolf) : LSPS_LinSolve(RES, JAC, precP, precE, tol_gmres, n_prec, n_iter, n_restart;f=linsolf)
			println("...LinSolve done  ||  Iter : ",Linsolveiter, " || rel_err : ",gmreserr)
			end
			linsol_time += _linsol_time
		   	#### Update and Print
			gmresnumcount += 1
			gmresitercount += Linsolveiter[1]
			prec=="CPR-LSPS" ? inneritercount+=Linsolveiter[2] : nothing
			LinearAlgebra.axpy!(-1.0, dg, psgrid_new) # Update step
			getresidual!(m, Δt, psgrid_new, psgrid_old, RES)
			norm_RES, norm_dg = norm(RES), norm(dg)
			@show norm_RES, norm_dg
		end
		griddiff = psgrid_new - psgrid_old
	        copyto!(psgrid_old, psgrid_new)
		push!(prod_rec, oil_production(m, Δt, psgrid_old))
       		record_p[:, steps] = [t_init+Δt; 2*sum(psgrid_old[1:2:end])/length(psgrid_old)]
		print_final(steps, t_init, Δt, psgrid_old, record_p, gmresitercount, itercount_div, inneritercount, griddiff)
		t_init += Δt
		itercount_total_c += gmresitercount
		itercount_total_d += itercount_div
		jac_time_total += jac_time
		linsol_time_total += linsol_time
		if (gmresnumcount < iternumtol)
			if (Δt>25.0&& Δt<=50.0) Δt=50.0 end
			if Δt<=25.0 Δt *= 2.0 end
		end
        	end
        	runtime += stepruntime
		print("Jacobian function time : ", round(jac_time, sigdigits=4))
       		println(" | Linsol time : ", round(linsol_time, sigdigits=4), " | % of linsol : ", round(100*linsol_time/stepruntime, sigdigits=4))
		println("Oil produced : ", Tuple(map(t->round(t, digits=3), sum(prod_rec))), " STB")
		println("Runtime on this step : ", round(stepruntime, sigdigits=6), " | Total Runtime : ", round(runtime, sigdigits=6), "sec", )
		if t_init>2000 break end
	end
	println("\nTotal Days : ",t_init-t_init_save ," | Total iteration(converge) : ", itercount_total_c, " | Total iteration(diverge) : ", itercount_total_d)
	println("Total Linsol time : ", linsol_time_total, "sec, Total Jacobian Call time : ", jac_time_total, "sec")
	println("Total Runtime : ", runtime, "sec")
	return psgrid_new, record_p, prod_rec
end

function print_final(steps, t_init, Δt, psgrid_old, record_p, gmresitercount, itercount_div, inneritercount, griddiff)
	println("CPR Stage 1 iteration : ", inneritercount, "| max dp, ds : ", (maximum(abs.(Array(griddiff[1:2:end]))), maximum(abs.(Array(griddiff[2:2:end])))))
        println("GMRES iteration(converge) : ",gmresitercount, " | GMRES iteration(diverge) : ", itercount_div)
	print("Min p : ", round(minimum(psgrid_old[1:2:end]), sigdigits=6), " | Max p : ", round(maximum(psgrid_old[1:2:end]), sigdigits=6))
	print("Min s : ", round(minimum(psgrid_old[2:2:end]), sigdigits=6), " | Max s : ", round(maximum(psgrid_old[2:2:end]), sigdigits=6))
        println("\nAvg p : ", record_p[2, steps], " | Total time : ", t_init+Δt, " Days")
end

function print_init(tol_relnorm, tol_gmres, n_restart, n_iter, CPR_prec, CPR_restart, CPR_iter, CPR_tol, prec, n_prec)
        println("CONVERGE_RELNORM : ", tol_relnorm, " | GMRES tol : ", tol_gmres," | GMRES restart : ", n_restart, " | GMRES maxiter : ", n_iter)
        if prec=="CPR-LSPS" println("Preconditioning : CPR-LSPS \n", "CPR-LSPS degree : ", CPR_prec, "| CPR restart : ", CPR_restart, " | CPR maxiter : ", CPR_iter, " | CPR gmres tol : ",CPR_tol) end
        if prec=="LSPS" println("Preconditioning : LSPS\n", "LSPS degree : ", n_prec) end
end

function CPR_LinSolve(RES, J, P, E, CPR_tol, CPR_iter, CPR_prec, CPR_restart, tol_gmres, n_iter, n_restart; f=fgmres)
    	Jp, Pp, Ep = CPR_Setup!(J, P, E)
    	inneriter = []
    	gmresresult = f(J, RES, n_restart;maxiter=n_iter, M=(t->CPR_LSPS(J, P, E, Jp, Pp, Ep, t, CPR_tol, CPR_iter, CPR_prec, CPR_restart, inneriter)), tol=tol_gmres)
	return gmresresult[1], gmresresult[3], (gmresresult[2], sum(inneriter)), gmresresult[4]
end
function LSPS_LinSolve(RES, J, P, E, tol_gmres, n_prec, n_iter, n_restart; f=fgmres)
	triLU!(P)
	gmresresult = f(J, RES, n_restart;maxiter=n_iter, M=(t->lsps_prec(P, E, n_prec, t)), tol=tol_gmres);
	return gmresresult[1], gmresresult[3], gmresresult[2], gmresresult[4]
end
