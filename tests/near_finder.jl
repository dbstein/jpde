push!(LOAD_PATH, pwd())

using Revise
using NearFinding
using Plots

"""
Function defining a star shaped object
Parameters:
    N:   number of points
    x:   x coordinate of center
    y:   y coordinate of center
    r:   nominal radius
    a:   amplitude of wobble, 0<a<1, smaller a is less wobbly
    f:   frequency - how many lobes are in the star
    rot: angle of rotation
"""
function star(N; x=0.0, y=0.0, r=1.0, a=0.5, f=3, rot=0.0)
	println(x, " ", y, " ", r, " ", a, " ", f, " ", rot)
	t = LinRange(0.0, 2π, N+1)[1:N]
	c = @. (x+im*y) + r*(1 + a*cos(f*(t-rot)))*exp(im*t)
	return real(c), imag(c)
end

Ng = 1001
Nb = 1000
xv = LinRange(-1.3, 1.3, 1001)
yv = LinRange(-1.3, 1.3, 1001)
bx, by = star(Nb, a=0.2, f=5)
d = 0.1

near, gi, closest = gridpoints_near_points(bx, by, xv, yv, d)

heatmap(xv, yv, near)

in_annulus, r, θ = gridpoints_near_curve(bx, by, xv, yv, d)

plt = heatmap(xv, yv, transpose(r))
# plot!(bx, by, color="white", legend=false)

plt = heatmap(xv, yv, transpose(θ))
# plot!(bx, by, color="white", legend=false)
