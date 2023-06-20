module Chebyshev

using Polynomials
using LinearAlgebra

export ChebyshevTPoints, chebyshev_ratio, inverse_chebyshev_ratio
export to_scaled_spaced, to_unscaled_space
export VandermondeMatrix, InverseVandermondeMatrix, VandermondeMatrixAndInverse
export DerivativeMatrix
export DecimationMatrix

"""
Provides chebyshev quadratures nodes
scaled to live on the interval [lb, ub], of specified order
The nodes are reversed from traditional chebyshev nodes
    (so that the lowest valued node comes first)
Returns:
    scaled nodes
"""
function ChebyshevTPoints(lb::T, ub::T, order::Integer) where T <: AbstractFloat
    return to_scaled_space.(lb, ub, ChebyshevTPoints(order, T))
end
function ChebyshevTPoints(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
	return cos.( (T(π)/2) .* (2 .* Vector{T}(order:-1:1) .- 1) ./ order )
end

################################################################################
# Convenience functions for transforming from the interval [lb, ub] to [-1, 1]

@inline function chebyshev_ratio(lb::T, ub::T) where T <: AbstractFloat
    return (ub-lb)/2
end
@inline function inverse_chebyshev_ratio(lb::T, ub::T) where T <: AbstractFloat
    return 2/(ub-lb)
end
@inline function to_scaled_space(lb::T, ub::T, x::T) where T <: AbstractFloat
    return (x+1)*chebyshev_ratio(lb, ub) + lb
end
@inline function to_unscaled_space(lb::T, ub::T, x::T) where T <: AbstractFloat
    return (x-lb)*inverse_chebyshev_ratio(lb, ub) - 1
end

################################################################################
# VandermondeMatrices

function VandermondeMatrix(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    return vander(ChebyshevT, ChebyshevTPoints(order, T), order-1)
end
function VandermondeMatrix(order::Integer, x::Vector{T}) where T <: AbstractFloat
    return vander(ChebyshevT, x, order-1)
end
function VandermondeMatrix(order::Integer, x::T) where T <: AbstractFloat
    return vander(ChebyshevT, [x], order-1)
end
function InverseVandermondeMatrix(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    return inv(VandermondeMatrix(order, T))
end
function VandermondeMatrixAndInverse(order::Integer, ::Type{T}=Float64) where T <: AbstractFloat
    V = VandermondeMatrix(order, T)
    return V, inv(V)
end

################################################################################
# Derivative Matrices

const _allowable = (:nodal, :coefficient)
@inline _check(x::Symbol) = x in _allowable

"""
Given a Chebyhsev operat OP which acts from OPIN --> OPOUT,
    create a new Chebyshev operator acting from REQIN --> REQOUT

For example, if OP acts from nodes --> nodes, then:
    NOP = toggle_modes(OP, :nodal, :nodal, :coefficient, :nodal)
returns a new operator NOP that acts from coefficients --> nodes

Any combination of OPIN, OPOUT, REQIN, and REQOUT are allowed,
    but each must be :nodal or :coefficient
"""
function toggle_modes(OP::Matrix{T}, OPIN, OPOUT, REQIN, REQOUT) where T <: AbstractFloat
    @assert _check(OPIN)
    @assert _check(OPOUT)
    @assert _check(REQIN)
    @assert _check(REQOUT)
    @assert OPIN in (:nodal, :coefficient)
    if OPIN == :coefficient && REQIN == :nodal
        OP = OP*InverseVandermondeMatrix(size(OP)[2], T)
    elseif OPIN == :nodal && REQIN == :coefficient
        OP = OP*VandermondeMatrix(size(OP)[2], T)
    end
    if OPOUT == :coefficient && REQOUT == :nodal
        OP = VandermondeMatrix(size(OP)[1])*OP
    elseif OPOUT == :nodal && REQOUT == :coefficient
        OP = InverseVandermondeMatrix(size(OP)[1])*OP
    end
    return OP
end

function DerivativeMatrix(
            order::Integer,
            D::Integer,
            ::Type{T}=Float64;
            in_type=:coefficient,
            out_type=:coefficient,
            scale_factor::T=one(T)
        ) where T <: AbstractFloat
    # construct mode --> mode derivative matrix
    DM = zeros(T, order-D, order)
    b = zeros(T, order)
    for i in eachindex(b)
        @. b *= zero(T)
        b[i] = one(T)
        w = derivative(ChebyshevT(b), D).coeffs
        DM[1:length(w), i] = w
    end
    DM .*= scale_factor^D
    return toggle_modes(DM, :coefficient, :coefficient, in_type, out_type)
end

################################################################################
# Decimation Operators

function DecimationMatrix(
        order_in::Integer,
        order_out::Integer,
        ::Type{T}=Float64;
        in_type=:coefficient,
        out_type=:coefficient
    ) where T <: AbstractFloat
    DM = zeros(T, order_out, order_in)
    DM[diagind(DM)] .= one(T)
    return toggle_modes(DM, :coefficient, :coefficient, in_type, out_type)
end

end