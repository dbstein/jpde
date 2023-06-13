module PeriodicInterpolaters

using FourierTools
using StaticArrays

export PeriodicInterpolater, PeriodicInterpolaterPrecompute
export DirectPeriodicInterpolate, DirectPeriodicInterpolate!

"""
1D Periodic Interpolation via Non-Uniform FFT

The two main structs here (PeriodicInterpoater and BatchedPeriodicInterpolater)
    can and should be merged together
    probably just a few re-interprets to get the job done
"""

@inbounds function φ(x::T, β::T) where T <: AbstractFloat
    x2 = x*x
    return x2 < one(T) ? exp(β*(sqrt(one(T)-x2)-one(T))) : zero(T)
end

"""
Reusable precomputations for a given tolerance ε and length N
"""
struct PeriodicInterpolaterPrecompute{T}
    ε::T
    iα::T
    β::T
    w2::Int64
    pk::Vector{Complex{T}}
end
function Base.show(PIP::PeriodicInterpolaterPrecompute)
    println("PeriodicInterpolater precomputation for N=", length(PIP.pk), " and ε=", PIP.ε, ".")
end
Base.print(PIP::PeriodicInterpolaterPrecompute) = show(PIP)
Base.display(PIP::PeriodicInterpolaterPrecompute) = show(PIP)
function PeriodicInterpolaterPrecompute(ε::T, N::Integer) where T <: AbstractFloat
    mod(N, 2) != 0 && error("N must be even.")
    w = 1 + ceil(Int64, log10(1/ε))
    β = 2.30 * w
    w2 = ceil(Int64, w/2)
    N2 = N ÷ 2
    x = LinRange(0, 2π, 2N+1)[1:2N]
    α = π*w/(2N)
    iα = 1/α
    ψ = fftshift(@. φ(iα*(x-π), β))
    ψh = fft(ψ)
    ψh = vcat(ψh[1:N2], ψh[end-N2+1:end]) # check these indeces!
    pk = @. 4π/(N*ψh)
    return PeriodicInterpolaterPrecompute{T}(ε, iα, β, w2, pk)
end

@inline function to_mat(f::AbstractVector)
    return Base.ReshapedArray(f, (1, length(f)), ())
end
@inline function to_mat(f::AbstractMatrix)
    return f
end

"""
Periodic Interpolater Class
"""
struct PeriodicInterpolater{T, TT, M}
    iα::T
    β::T
    w2::Int64
    padf::Matrix{TT}
    scalar::Bool
end
function Base.show(PI::PeriodicInterpolater{T, TT}) where {T, TT}
    if PI.scalar
        println("1D PeriodicInterpolater for single function with type ", TT, ".")
    else
        println("1D Batched PeriodicInterpolater for ", getM(PI), " functions with type ", TT, ".")
    end
end
Base.print(PI::PeriodicInterpolater) = show(PI)
Base.display(PI::PeriodicInterpolater) = show(PI)
function getN(f::AbstractVector)
    return length(f)
end
function getN(f::AbstractMatrix)
    return size(f)[2]
end
function getM(PI::PeriodicInterpolater)
    return size(PI.padf)[1]
end
function PeriodicInterpolater(f::AbstractVecOrMat{TT}, ε::T, PIP::PeriodicInterpolaterPrecompute{T}) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}}
    scalar = typeof(f) <: AbstractVector
    f = to_mat(f)
    fh = fft(f, 2) # which dimension should this go along?
    M = size(fh)[1]
    N = size(fh)[2]
    N2 = N ÷ 2
    fh_adj = transpose(PIP.pk) .* fh .* (N/(2π))
    pad_fh_adj = zeros(Complex{T}, M, 2N)
    pad_fh_adj[:, 1:N2] =  fh_adj[:, 1:N2]
    pad_fh_adj[:, end-N2+1:end] = fh_adj[:, end-N2+1:end]
    padf = ifft(pad_fh_adj, 2)
    if TT <: Real
        return PeriodicInterpolater{T, T, M}(PIP.iα, PIP.β, PIP.w2, real.(padf), scalar)
    else
        return PeriodicInterpolater{T, TT, M}(PIP.iα, PIP.β, PIP.w2, padf, scalar)
    end
end
function PeriodicInterpolater(f::AbstractVecOrMat{TT}, ε::T) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}}
    return PeriodicInterpolater(f, ε, PeriodicInterpolaterPrecompute(ε, getN(f)))
end
function (PI::PeriodicInterpolater{T, TT, M})(x::AbstractVector{T}) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}, M}
    out = PI.scalar ? Vector{TT}(undef, length(x)) : Matrix{TT}(undef, (M, length(x)))
    convolve_v!(to_mat(out), x, PI.padf, PI.iα, PI.w2, PI.β, Val(M))
    return out
end
function (PI::PeriodicInterpolater{T, TT, M})(out::AbstractVecOrMat{TT}, x::AbstractVector{T}) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}, M}
    convolve_v!(to_mat(out), x, PI.padf, PI.iα, PI.w2, PI.β, Val(M))
    return out
end
function (PI::PeriodicInterpolater{T, TT, M})(x::T) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}, M}
    return convolve_1(x, PI.padf, PI.iα, PI.w2, PI.β, Val(M))
end

"""
Batched Convolution Functions
"""

function convolve_1(x::T, big_y::AbstractMatrix{TT}, iα::T, w2::Integer, β::T, ::Val{M}) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}, M}
    N = size(big_y)[2]
    h = 2π/N
    out = @SVector zeros(TT, M)
    ind = mod(floor(Int64, x / h), N) + 1
    min_ind = ind - w2 # check these indeces
    max_ind = ind + w2 + 1
    xm = mod(x, 2π) # maybe slightly fragile --- mod2pi(x) failed at 2π because mod2pi(2π)->2pπ, but ind was 0
    # reinterpret big_y as an array of SArrays
    bigyr = reinterpret(SVector{M, TT}, big_y)
    @fastmath for j in min_ind:max_ind
        z = iα*(xm - j*h)
        @inbounds out += φ(z, β)*bigyr[mod(j, N)+1]
    end
    return out
end
function convolve_v!(out::AbstractMatrix{TT}, xs::AbstractVector{T}, big_y::Matrix{TT}, iα::T, w2::Integer, β::T, ::Val{M}) where {T <: AbstractFloat, TT <: Union{T, Complex{T}}, M}
    outr = reinterpret(SVector{M, TT}, out)
    Threads.@threads for i in eachindex(xs)
        @inbounds outr[i] = convolve_1(xs[i], big_y, iα, w2, β, Val(M))
    end
    return out
end

"""
Direct Interpolation Functions
(really just for testing purposes, though in small cases may be faster)
"""

@inbounds function DirectPeriodicInterpolate(x::T, fh::Vector{Complex{T}}) where T <: AbstractFloat
    sz = length(fh)
    sz2 = sz ÷ 2
    out = zero(T)
    xx = exp(im*x)
    ixx = 1.0/xx
    xh1 = 1.0
    xh2 = ixx
    @inbounds for i in 1:sz2
        out += xh1*fh[i] + xh2*fh[sz+1-i]
        xh1 *= xx
        xh2 *= ixx 
    end
    return out/sz
end
function DirectPeriodicInterpolate(x::AbstractVector{T}, fh::Vector{Complex{T}}) where T <: AbstractFloat
    out = zeros(Complex{T}, length(x))
    return DirectPeriodicInterpolate!(out, x, fh)
end
function DirectPeriodicInterpolate!(out::AbstractVector{Complex{T}}, x::AbstractVector{T}, fh::Vector{Complex{T}}) where T <: AbstractFloat
    Threads.@threads for i in eachindex(x, out)
        @inbounds out[i] = DirectPeriodicInterpolate(x[i], fh)
    end
    return out
end

end