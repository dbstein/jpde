push!(LOAD_PATH, pwd())

using Revise
using AnnularSolvers
using BoundaryShapes
using CurveGeometry
using DoubleFloats

Nb = 200
M = 16
T = Float64
helmholtz_k = one(T)
bx, by = star(Nb, T; a=T(2//10), f=5);

AAG = ApproximateAnnularGeometry( Nb, M, T(1//10), one(T) );
RAG = RealAnnularGeometry(bx, by, AAG);
AMHS = AnnularModifiedHelmholtzSolver(AAG; helmholtz_k=helmholtz_k);

k = 2T(π)/3
solution_func(x, y) = exp(sin(k*x))*sin(k*y)
force_func(x, y) = helmholtz_k^2*solution_func(x, y)-k^2*exp(sin(k*x))*sin(k*y)*(cos(k*x)^2-sin(k*x)-one(T))

force = force_func.(RAG.radial_x, RAG.radial_y);
asol = solution_func.(RAG.radial_x, RAG.radial_y);
ibc = solution_func.(RAG.interface_x, RAG.interface_y);
ubc = solution_func.(RAG.boundary_x, RAG.boundary_y);

u = AMHS(RAG, force, ubc, ibc; verbose=true);
err = maximum(abs.(u-asol));
print("Error is: ", err)
