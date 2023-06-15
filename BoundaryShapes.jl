module BoundaryShapes

export star

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
			) where T <: AbstractFloat
	t = LinRange(zero(T), 2T(π), N+1)[1:N]
	c = @. (x+im*y) + r*(1 + a*cos(f*(t-rot)))*exp(Complex{T}(im)*t)
	return real(c), imag(c)
end

end