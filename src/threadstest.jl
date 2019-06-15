include("test_without_comp.jl")
using BenchmarkTools
R1 = loaddata("step50.h5", 2)[1];
J, P, E, diagW = GenericReservoir.getjacobian2(model_gpu, 5.12, R1, R1);
RES = GenericReservoir.getresidual(model_gpu, 5.12, R1, R1);
Jp, Pp, Ep, W = GenericReservoir.CPR_Setup!(J, P, E);
rp = (W*RES)[1:2:end];
x = zero(rp);
X = zero(RES);

for threads in [16, 32, 64, 128], divnum in [85, 170, 340, 510, 680]
	@show (threads, divnum)
	@btime CuArrays.@sync GenericReservoir.block_execute!(GenericReservoir.tridiagonal_LU_solve_block!, $threads, 1122000, $divnum, P.diags[1].second, P.diags[2].second, P.diags[3].second, P.diags[4].second, P.diags[5].second, P.diags[6].second, P.diags[7].second, copy(RES), X);
end
