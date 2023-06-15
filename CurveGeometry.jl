module CurveGeometry

using AbstractFFTs
using GenericFFT

export compute_speed, compute_curvature, CurveDescription

struct CurveDescription{T}
    x::Vector{T}
    y::Vector{T}
    τx::Vector{T}
    τy::Vector{T}
    nx::Vector{T}
    ny::Vector{T}
    speed::Vector{T}
    curvature::Vector{T}
end
function CurveDescription(
            x::AbstractVector{T},
            y::AbstractVector{T}
        ) where T <: AbstractFloat
    n = length(x)
    k = rfftfreq(n, T(n))
    xh = rfft(x)
    yh = rfft(y)
    xp  = irfft(im.*k .* xh, n)
    yp  = irfft(im.*k .* yh, n)
    xpp = irfft(-k.^2 .* xh, n)
    ypp = irfft(-k.^2 .* yh, n)
    speed = hypot.(xp, yp)
    ispeed = one(T)./speed
    τx = xp.*ispeed
    τy = yp.*ispeed
    nx = τy
    ny = -τx
    curvature = @. (ypp*xp - xpp*yp)*ispeed^3
    return CurveDescription{T}(copy(x), copy(y), τx, τy, nx, ny, speed, curvature)
end
@inline Base.length(CD::CurveDescription) = length(CD.x)
@inline Base.eltype(CD::CurveDescription{T}) where T = T
function Base.show(CD::CurveDescription)
    println("CurveDescription for curve with type: ", eltype(CD), "; and n: ", length(CD), ".")
end
Base.print(CD::CurveDescription) = show(CD)
Base.display(CD::CurveDescription) = show(CD)


"""
Computes the speed of a closed plane curve (x(s), y(s))
given points x(s_i), y(s_i)
the final point is assumed to be not repeated

The curve is assumed parametrized on [0, 2π]
"""
function compute_speed(
            x::AbstractVector{T}, 
            y::AbstractVector{T}, 
            k::AbstractVector{T}
        ) where T <: AbstractFloat
    n = length(x)
    xp = irfft(im.*k.*rfft(x), n)
    yp = irfft(im.*k.*rfft(y), n)
    return hypot.(xp, yp)
end
function compute_speed(
            x::AbstractVector{T},
            y::AbstractVector{T}
        ) where T <: AbstractFloat
    n = length(x)
    return compute_speed(x, y, rfftfreq(n, T(n)))
end
function compute_speed(c::AbstractVector{T}, ik::AbstractVector{T}) where T <: Complex
    return abs.(ifft(ik .* fft(c)))
end
function compute_speed(c::AbstractVector{Complex{T}}) where T <: AbstractFloat
    n = length(c)
    return compute_speed(x, y, im*fftfreq(n, T(n)))
end

"""
Computes the curvature of a closed plane curve (x(s), y(s))
given points x(s_i), y(s_i)
the final point is assumed to be not repeated
"""
function compute_curvature(
            x::AbstractVector{T}, 
            y::AbstractVector{T},
            k::AbstractVector{T}
        ) where T <: AbstractFloat
    n = length(x)
    xh = rfft(x)
    yh = rfft(y)
    xp  = irfft(im.*k .* xh, n)
    yp  = irfft(im.*k .* yh, n)
    xpp = irfft(-k.^2 .* xh, n)
    ypp = irfft(-k.^2 .* yh, n)
    speed = hypot.(xp, yp)
    curvature = @. (ypp*xp - xpp*yp)/speed^3
    return curvature, speed
end
function compute_curvature(
            x::AbstractVector{T}, 
            y::AbstractVector{T}
        ) where T <: AbstractFloat
    n = length(x)
    return compute_curvature(x, y, rfftfreq(n, T(n)))
end
function compute_curvature(
            c::AbstractVector{Complex{T}},
            ik::AbstractVector{Complex{T}}
        ) where T <: AbstractFloat
    ch = fft(c)
    cp  = ifft(@. ik * ch)
    cpp = ifft(@. ik^2 * ch)
    speed = abs.(cp)
    normal_c = -im*cp/speed
    curvature = @. real(conj(cpp)*cp*im)/speed^3
    return curvature, speed
end
function compute_curvature(
            c::AbstractVector{Complex{T}}
        ) where T <: AbstractFloat
    n = length(c)
    return compute_curvature(c, ik = im*rfftfreq(n, T(n)))
end

end
