push!(LOAD_PATH, pwd())

using Revise
using NearFinding
using DoubleFloats
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
function star(	N::Integer,
				::Type{T}=Float64; 
				x::T=zero(T),
				y::T=zero(T),
				r::T=one(T),
				a::T=T(1//2),
				f::Integer=3,
				rot::T=zero(T)
			) where T <: Real
	t = LinRange(zero(T), 2T(π), N+1)[1:N]
	c = @. (x+im*y) + r*(1 + a*cos(f*(t-rot)))*exp(Complex{T}(im)*t)
	return real(c), imag(c)
end

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
