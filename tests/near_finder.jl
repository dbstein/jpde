push!(LOAD_PATH, pwd())

using Revise
using NearFinding
using BoundaryShapes
using DoubleFloats
using Plots

Ng = 101
Nb = 100
T = Float64
xv = LinRange(T(-1.3), T(1.3), Ng)
yv = LinRange(T(-1.3), T(1.3), Ng)
bx, by = star(Nb, T; a=T(0.2), f=5)
d = T(0.1)

near, gi, closest = gridpoints_near_points(bx, by, xv, yv, d)

heatmap(xv, yv, near)

in_annulus, r, θ = gridpoints_near_curve(bx, by, xv, yv, d)

plt = heatmap(xv, yv, transpose(r))
plot!(bx, by, color="white", legend=false)

plt = heatmap(xv, yv, transpose(θ))
plot!(bx, by, color="white", legend=false)
