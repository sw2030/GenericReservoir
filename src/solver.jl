function ReservoirSolve(m::Reservoir_Model{T}, t_init, Δt, g_guess, n_steps;prec="CPR-LSPS", tol_relnorm=1e-3, tol_gmres=1e-2, n_restart=10, n_iter=10, n_prec=7, step_init=0, CPR_tol=1e-1, CPR_iter=5, CPR_prec=(15, 15), CPR_restart=15, iternumtol=7, nl_tol=10, mg_iter=3, linsolf=fgmres) where {T}

   	grid_old = copy(g_guess)
   	grid_new = copy(grid_old)
   	runtime = 0.0
	record_p   = zeros(2, n_steps)
    prod_rec = []
   	itercount_total_c, itercount_total_d, t_init_save, jac_time_total, linsol_time_total, JAC_FUNC_CALL = 0, 0, t_init, 0.0, 0.0, 0
	#### Print Arguments
	print_init(tol_relnorm, tol_gmres, n_restart, n_iter, CPR_prec, CPR_restart, CPR_iter, CPR_tol, prec, n_prec)
	
	#### Time stepping start
    for steps in 1:n_steps
       	stepruntime = @elapsed CuArrays.@sync begin
		
		#### Initialize Each Steps
        RES = getresidual(m, Δt, grid_new, grid_old)
       	norm_RES_save = norm(RES)
       	norm_RES = norm_RES_save
       	println("\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
       	gmresnumcount, gmresitercount, itercount_div, inneritercount, j_t, linsol_time = 0, 0, 0, 0, 0.0, 0.0

       	while(norm_RES/norm_RES_save>tol_relnorm)
           	#### In case it is Divergine
    		if ((norm_RES>1e5 && steps>1) || gmresnumcount >= nl_tol || (gmresnumcount>6 && norm_RES>1.0e4)) ### These are the conditions for reducing Δt (Not converged)
               		copyto!(grid_new, grid_old)
               		Δt *= 0.5
               		itercount_div += gmresitercount
               		gmresnumcount, gmresitercount = 0, 0
               		getresidual!(m, Δt, grid_new, grid_old, RES)
               		norm_RES = norm(RES)
               		println("\nNot Converged, Δt reduced... Δt : ",Δt*2.0, "->", Δt,"\n\nSTEP ", steps+step_init, " | norm_RES : ", norm_RES, " | Δt : ",Δt)
			end

            if prec=="LSPS"
                dg, j_t, l_t, it_count = LinSolve_LSPS(m, Δt, RES, grid_old, grid_new, (n_restart, n_iter, tol_gmres), n_prec;linsolve=linsolf)
            elseif prec=="CPR-LSPS"
                dg, j_t, l_t, it_count = LinSolve_CPR_LSPS(m, Δt, RES, grid_old, grid_new, 
                                                                (n_restart, n_iter, tol_gmres), 
                                                                (CPR_restart, CPR_iter, CPR_tol, CPR_prec))
            elseif prec=="CPR-MG"
                dg, j_t, l_t, it_count = LinSolve_CPR_MG(m, Δt, RES, grid_old, grid_new,
                                                                (n_restart, n_iter, tol_gmres), 
                                                                (CPR_restart, CPR_iter, CPR_tol, n_prec, mg_iter))        
            end
            
            JAC_FUNC_CALL += 1
			linsol_time += l_t
            gmresnumcount += 1
            gmresitercount += it_count[1]
            inneritercount += it_count[2]
		   	
            #### Update and Print
			LinearAlgebra.axpy!(-1.0, dg, grid_new) # Update step
			getresidual!(m, Δt, grid_new, grid_old, RES)
    		norm_RES, norm_dg = norm(RES), norm(dg)
			@show norm_RES, norm_dg
		end
		griddiff = grid_new - grid_old
	    copyto!(grid_old, grid_new)
		push!(prod_rec, oil_production(m, Δt, grid_old))
    	record_p[:, steps] = [t_init+Δt; 2*sum(grid_old[1:2:end])/length(grid_old)]
		print_final(steps, t_init, Δt, grid_old, record_p, gmresitercount, itercount_div, inneritercount, griddiff)
		t_init += Δt
		itercount_total_c += gmresitercount
		itercount_total_d += itercount_div
		jac_time_total += j_t
		linsol_time_total += linsol_time
		if (gmresnumcount < iternumtol)
			if (Δt>25.0&& Δt<=50.0) Δt=50.0 end
			if Δt<=25.0 Δt *= 2.0 end
		end
        end
        runtime += stepruntime
		print("Jacobian function time : ", round(j_t, sigdigits=4))
       	println(" | Linsol time : ", round(linsol_time, sigdigits=4), " | % of linsol : ", round(100*linsol_time/stepruntime, sigdigits=4))
		println("Oil produced : ", Tuple(map(t->round(t, digits=3), sum(prod_rec))), " STB")
		println("Runtime on this step : ", round(stepruntime, sigdigits=6), " | Total Runtime : ", round(runtime, sigdigits=6), "sec", )
		if t_init>2000 break end
	end
	println("\nTotal Days : ",t_init-t_init_save ," | Total iteration(converge) : ", itercount_total_c, " | Total iteration(diverge) : ", itercount_total_d)
	println("Total Linsol time : ", linsol_time_total, "sec, Total Jacobian Call time : ", jac_time_total, "sec\nTotal Jacobian Function call : ", JAC_FUNC_CALL)
	println("Total Runtime : ", runtime, "sec")
	return grid_new, prod_rec
end

function print_final(steps, t_init, Δt, grid_old, record_p, gmresitercount, itercount_div, inneritercount, griddiff)
	println("CPR Stage 1 iteration : ", inneritercount, "| max dp, ds : ", (maximum(abs.(griddiff[1:2:end])), maximum(abs.(griddiff[2:2:end]))))
    println("GMRES iteration(converge) : ",gmresitercount, " | GMRES iteration(diverge) : ", itercount_div)
	print("Min p : ", round(minimum(grid_old[1:2:end]), sigdigits=6), " | Max p : ", round(maximum(grid_old[1:2:end]), sigdigits=6), " | ")
	print("Min s : ", round(minimum(grid_old[2:2:end]), sigdigits=6), " | Max s : ", round(maximum(grid_old[2:2:end]), sigdigits=6))
    println("\nAvg p : ", record_p[2, steps], " | Total time : ", t_init+Δt, " Days")
end

function print_init(tol_relnorm, tol_gmres, n_restart, n_iter, CPR_prec, CPR_restart, CPR_iter, CPR_tol, prec, n_prec)
    println("CONVERGE_RELNORM : ", tol_relnorm, " | GMRES tol : ", tol_gmres," | GMRES restart : ", n_restart, " | GMRES maxiter : ", n_iter)
    if prec=="CPR-LSPS" println("Preconditioning : CPR-LSPS \n", "CPR-LSPS degree : ", CPR_prec, "| CPR restart : ", CPR_restart, " | CPR maxiter : ", CPR_iter, " | CPR gmres tol : ",CPR_tol) end
    if prec=="LSPS" println("Preconditioning : LSPS\n", "LSPS degree : ", n_prec) end
end


