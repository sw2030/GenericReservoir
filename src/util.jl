function testsolve(m, g)
    println("===================Test run for Compiling===================")
    GenericReservoir.ReservoirSolve(m, 0.0, 0.001, g, 1;prec="CPR-LSPS");
    GenericReservoir.ReservoirSolve(m, 0.0, 0.001, g, 1;prec="LSPS"); ## Compilation
    println("============================================================")
end
function savedata(d, fname, Num)
    h5write(string(fname, ".h5"), string("Data/",Num), Array(d[1]))
    h5write(string(fname, ".h5"), string("Log/",Num), d[2])
    h5write(string(fname, ".h5"), string("prod/",Num), [d[3][j][i] for i in 1:4, j in 1:length(d[3])])
    nothing
end
function loaddata(fname, Num)
    d = h5read(string(fname, ".h5"), string("Data/",Num))
    logd = h5read(string(fname, ".h5"), string("Log/",Num))
    prod = h5read(string(fname, ".h5"), string("prod/",Num))
    return CuArray(d), logd, prod
end
