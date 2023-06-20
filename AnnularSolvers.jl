module AnnularSolvers

using CurveGeometry
using AbstractFFTs
using GenericFFT
using Chebyshev
using LinearSolve
using SciMLOperators
using LinearAlgebra

export ChebyshevOperators, ApproximateAnnularGeometry, RealAnnularGeometry
export AnnularModifiedHelmholtzSolver
export scalar_laplacian

# note: can probably improve performance by making these static...
# note: also, can improve usability by removing rat from constructions
#   and applying on the fly later
struct ChebyshevOperators{T}
    M::Int64
    rat::T
    # Vandermonde Matrices
    V0::Matrix{T}
    V1::Matrix{T}
    V2::Matrix{T}
    # inverse Vandermonde Matrices
    IV0::Matrix{T}
    IV1::Matrix{T}
    IV2::Matrix{T}
    # differentiation matrices
    DM01::Matrix{T}
    DM12::Matrix{T}
    # boundary condition matrices
    IBCD::Matrix{T} 
    OBCD::Matrix{T}
    IBCN::Matrix{T} 
    OBCN::Matrix{T}
    # order changing operators
    RR01::Matrix{T}
    RR12::Matrix{T}
    RR02::Matrix{T}
    RR10::Matrix{T}
end
function ChebyshevOperators( M::Integer, rat::T ) where T <: AbstractFloat
    # Vandermonde Matrices
    V0, IV0 = VandermondeMatrixAndInverse(M)
    V1, IV1 = VandermondeMatrixAndInverse(M-1)
    V2, IV2 = VandermondeMatrixAndInverse(M-2)
    # Coefficient --> Coefficient Derivative Matrices
    DM01 = DerivativeMatrix(M, 1, T; in_type=:nodal, out_type=:nodal, scale_factor=rat)
    DM12 = DerivativeMatrix(M-1, 1, T; in_type=:nodal, out_type=:nodal, scale_factor=rat)
    # Boundary Condition Matrices
    IBCD = VandermondeMatrix(M, one(T))*IV0
    OBCD = VandermondeMatrix(M, -one(T))*IV0
    DM = DerivativeMatrix(M, 1, T; in_type=:nodal, out_type=:coefficient, scale_factor=rat)
    IBCN = VandermondeMatrix(M-1, one(T))*DM
    OBCN = VandermondeMatrix(M-1, -one(T))*DM
    # Rank Reduction Operators
    RR01 = DecimationMatrix(M, M-1, T; in_type=:nodal, out_type=:nodal)
    RR12 = DecimationMatrix(M-1, M-2, T; in_type=:nodal, out_type=:nodal)
    RR02 = DecimationMatrix(M, M-2, T; in_type=:nodal, out_type=:nodal)
    RR10 = DecimationMatrix(M-1, M, T; in_type=:nodal, out_type=:nodal)
    return ChebyshevOperators{T}(M, rat, V0, V1, V2, IV0, IV1, IV2, DM01, DM12, IBCD, OBCD, IBCN, OBCN, RR01, RR12, RR02, RR10)
end
function Base.show(CO::ChebyshevOperators{T}) where {T}
    println("Chebyshev Operators for Annular Solvers")
    println("...with type ", T, " and order ", CO.M, ".")
end
Base.print(CO::ChebyshevOperators) = show(CO)
Base.display(CO::ChebyshevOperators) = show(CO)

"""
Approximate Annular Geometry for solving PDE in annular regions
n: number of discrete points in tangential direction
M: number of chebyshev modes in radial direction
width: width of radial region
approx_r: approximate radius of annulus
"""
struct ApproximateAnnularGeometry{T}
    n::Int64
    M::Int64
    approx_r::T
    width::T
    radial_h::T
    tangent_h::T
    n2::Int64
    k::Vector{T}
    ik::Vector{Complex{T}}
    rv0::Vector{T}
    rv1::Vector{T}
    rv2::Vector{T}
    ratio::T
    approx_ψ0::Vector{T}
    approx_ψ1::Vector{T}
    approx_ψ2::Vector{T}
    approx_inv_ψ0::Vector{T}
    approx_inv_ψ1::Vector{T}
    approx_inv_ψ2::Vector{T}
    CO::ChebyshevOperators{T}
end
function ApproximateAnnularGeometry(
            n::Integer,
            M::Integer,
            width::T,
            approx_r::T
        ) where T <: AbstractFloat

    radial_h = width/M
    tangent_h = 2T(π)/n
    n2 = n ÷ 2
    k = fftfreq(n, T(n))
    # note the python code strips out the nyquist mode
    # lets try to get something that works without doing that
    ik = Complex{T}(im)*k
    # r grids
    rv0 = ChebyshevTPoints(-width, zero(T), M)
    rv1 = ChebyshevTPoints(-width, zero(T), M-1)
    rv2 = ChebyshevTPoints(-width, zero(T), M-2)
    ratio = inverse_chebyshev_ratio(-width, zero(T))
    # coordinate transfromations
    approx_ψ0 = approx_r .+ rv0
    approx_ψ1 = approx_r .+ rv1
    approx_ψ2 = approx_r .+ rv2
    approx_inv_ψ0 = one(T) ./ approx_ψ0
    approx_inv_ψ1 = one(T) ./ approx_ψ1
    approx_inv_ψ2 = one(T) ./ approx_ψ2
    # Chebyshev Operators
    CO = ChebyshevOperators(M, ratio)
    return ApproximateAnnularGeometry{T}(n, M, approx_r, width, radial_h, tangent_h,
            n2, k, ik, rv0, rv1, rv2, ratio, approx_ψ0, approx_ψ1, approx_ψ2,
            approx_inv_ψ0, approx_inv_ψ1, approx_inv_ψ2, CO)
end
@inline Base.reshape(AAG::ApproximateAnnularGeometry, f::AbstractVector) = reshape(f, AAG.M, AAG.n)
@inline Base.reshape(AAG::ApproximateAnnularGeometry, f::AbstractMatrix) = f
function Base.show(AAG::ApproximateAnnularGeometry{T}) where {T}
    println("ApproximateAnnularGeometry for Annular Solvers")
    println("...with type ", T, ", n = ", AAG.n, ", and radial order ", AAG.M, ".")
end
Base.print(AAG::ApproximateAnnularGeometry) = show(AAG)
Base.display(AAG::ApproximateAnnularGeometry) = show(AAG)

# this should also construct the x, y matrices
# and the lower and upper curves
struct RealAnnularGeometry{T}
    CD::CurveDescription{T}
    boundary_x::Vector{T}
    boundary_y::Vector{T}
    interface_x::Vector{T}
    interface_y::Vector{T}
    radial_x::Matrix{T}
    radial_y::Matrix{T}
    ψ0::Matrix{T}
    ψ1::Matrix{T}
    ψ2::Matrix{T}
    inv_ψ0::Matrix{T}
    inv_ψ1::Matrix{T}
    inv_ψ2::Matrix{T}
    DR_ψ2::Matrix{T}
    ipsi_DR_ipsi_DT_psi2::Matrix{T}
    ipsi_DT_ipsi_DR_psi2::Matrix{T}
end
function RealAnnularGeometry(
            bx::Vector{T},
            by::Vector{T},
            AAG::ApproximateAnnularGeometry{T}
        ) where T <: AbstractFloat
    return RealAnnularGeometry(CurveDescription(bx, by), AAG)
end
function RealAnnularGeometry(
            CD::CurveDescription{T},
            AAG::ApproximateAnnularGeometry{T}
        ) where T <: AbstractFloat
    # get boundary / interface
    boundary_x = CD.x
    boundary_y = CD.y
    # should this use a difference width than AAG?  probably
    interface_x = CD.x .- AAG.width*CD.nx
    interface_y = CD.y .- AAG.width*CD.ny
    # get radial grids
    radial_x = transpose(boundary_x) .+ AAG.rv0.*transpose(CD.nx)
    radial_y = transpose(boundary_y) .+ AAG.rv0.*transpose(CD.ny)
    dt_curvature = real.(ifft(fft(CD.curvature).*AAG.ik))
    speed = transpose(CD.speed)
    curvature = transpose(CD.curvature)
    dt_curvature = transpose(dt_curvature)
    ψ0 = speed .* (1 .+ AAG.rv0 .* curvature)
    ψ1 = speed .* (1 .+ AAG.rv1 .* curvature)
    ψ2 = speed .* (1 .+ AAG.rv2 .* curvature)
    inv_ψ0 = one(T) ./ ψ0
    inv_ψ1 = one(T) ./ ψ1
    inv_ψ2 = one(T) ./ ψ2
    DR_ψ2 = speed .* curvature .* ones(T, AAG.M)
    denom2 = speed .* (1 .+ AAG.rv2 .* curvature).^3
    idenom2 = one(T) ./ denom2
    # these are what i think it should be? need to check computation
    # ipsi_DR_ipsi_DT_psi2 = (curvature-dt_curvature)*idenom2
    # ipsi_DT_ipsi_DR_psi2 = -dt_curvature*idenom2
    # these are what work...
    ipsi_DR_ipsi_DT_psi2 = dt_curvature .* idenom2
    ipsi_DT_ipsi_DR_psi2 = dt_curvature .* idenom2
    return RealAnnularGeometry{T}(CD, boundary_x, boundary_y, interface_x, interface_y,
                radial_x, radial_y, ψ0, ψ1, ψ2, inv_ψ0, inv_ψ1, inv_ψ2,
                DR_ψ2, ipsi_DR_ipsi_DT_psi2, ipsi_DT_ipsi_DR_psi2)
end
function Base.show(RAG::RealAnnularGeometry{T}) where {T}
    println("RealAnnularGeometry for Annular Solvers with type ", T, ".")
end
Base.print(RAG::RealAnnularGeometry) = show(RAG)
Base.display(RAG::RealAnnularGeometry) = show(RAG)

function fm(fh, g)
    return fft(ifft(fh, 2).*g, 2)
end

function scalar_laplacian(
            AAG::ApproximateAnnularGeometry{T},
            RAG::RealAnnularGeometry{T},
            uh::AbstractMatrix{Complex{T}}
        ) where T <: AbstractFloat
    CO = AAG.CO
    ik = transpose(AAG.ik)
    uh_t = CO.RR01 * (uh.*ik)
    uh_tt = CO.RR12 * (fm(uh_t, RAG.inv_ψ1) .* ik)
    uh_rr = CO.DM12 * fm(CO.DM01 * uh, RAG.ψ1)
    return fm(uh_rr+uh_tt, RAG.inv_ψ2)
end

"""
REPLICATING NUMPY FUNCTIONALITY, SHOULD BE REPLACED

Specialized interface to the numpy.dot function
This assumes that A and B are both 2D arrays (in practice)
When A or B are represented by 1D arrays, they are assumed to reprsent
    diagonal arrays
This function then exploits that to provide faster multiplication
"""
function fast_dot(M1::AbstractVector, M2::AbstractVecOrMat)
    return M1 .* M2
end
function fast_dot(M1::AbstractVecOrMat, M2::AbstractVector)
    return M1 .* transpose(M2)
end
function fast_dot(M1::AbstractMatrix, M2::AbstractMatrix)
    return M1*M2
end

struct AnnularModifiedHelmholtzSolver{T}
    helmholtz_k::T
    AAG::ApproximateAnnularGeometry{T}
    KINVS::Vector{Matrix{T}}
    ia::T
    ib::T
    oa::T
    ob::T
end
function AnnularModifiedHelmholtzSolver(
        AAG::ApproximateAnnularGeometry{T};
        helmholtz_k::T=one(T),
        ia::T=one(T),
        ib::T=zero(T),
        oa::T=one(T),
        ob::T=zero(T)
    ) where T <: AbstractFloat
    CO = AAG.CO
    M = AAG.M
    NB = M*AAG.n
    shape = (M, AAG.n)
    aψ1 =  AAG.approx_ψ1
    aiψ1 = AAG.approx_inv_ψ1
    aiψ2 = AAG.approx_inv_ψ2
    KINVS = Matrix{T}[]
    for i in 1:AAG.n
        K = zeros(T, M, M)
        LL = (fast_dot(aiψ2, fast_dot(CO.DM12, fast_dot(aψ1, CO.DM01))) -
                 fast_dot(ones(T, AAG.M-2)*AAG.k[i]^2, fast_dot(CO.RR12, fast_dot(aiψ1, CO.RR01))))
        K[1:M-2, :] = @. helmholtz_k^2 * CO.RR02 - LL
        K[end-1, :] = @. ia*CO.IBCD + ib*CO.IBCN
        K[end, :] = @. oa*CO.OBCD + ob*CO.OBCN
        push!(KINVS, inv(K))
     end
     return AnnularModifiedHelmholtzSolver{T}(helmholtz_k, AAG, KINVS, ia, ib, oa, ob)
end
function Base.show(AMHS::AnnularModifiedHelmholtzSolver{T}) where {T}
    println("AnnularModified HelmholtzSolverwith type ", T, ".")
    println("Underlying AAG is:")
    show(AMHS.AAG)
    println("Modified Helmholtz Parameter is: ", AMHS.helmholtz_k)
    println("Inner Boundary Condition parmaters ia=", AMHS.ia, "; ib=", AMHS.ib)
    println("Outer Boundary Condition parmaters oa=", AMHS.oa, "; ob=", AMHS.ob)
end
Base.print(AMHS::AnnularModifiedHelmholtzSolver) = show(AMHS)
Base.display(AMHS::AnnularModifiedHelmholtzSolver) = show(AMHS)

struct MyPreconditioner{T}
    KINVS::Vector{Matrix{T}}
    M::Int64
    n::Int64
end
function MyPreconditioner(KINVS::Vector{Matrix{T}}, M, n) where T
    return MyPreconditioner{T}(KINVS, M, n)
end
Base.reshape(MP::MyPreconditioner, B::AbstractVector) = reshape(B, MP.M, MP.n)
Base.reshape(MP::MyPreconditioner, B::AbstractMatrix) = B
function LinearAlgebra.ldiv!(Y::AbstractVector{Complex{T}}, A::MyPreconditioner{T}, B::AbstractVector{Complex{T}}) where T
    B = reshape(A, B)
    Y = reshape(A, Y)
    for i in 1:A.n
        view(Y, :, i) .= A.KINVS[i] * view(B, :, i)
    end
    return Y
end
function LinearAlgebra.ldiv!(A::MyPreconditioner{T}, B::AbstractVector{Complex{T}}) where T
    B = reshape(A, B)
    for i in 1:A.n
        mul!(A.KINVS[i], view(B, :, i))
    end
    return B
end

function (AMHS::AnnularModifiedHelmholtzSolver{T})(
                RAG::RealAnnularGeometry{T},
                f::AbstractMatrix{T},
                ig::AbstractVector{T},
                og::AbstractVector{T};
                ia::T=one(T),
                ib::T=zero(T),
                oa::T=one(T),
                ob::T=zero(T),
                atol::T=100eps(T),
                rtol::T=100eps(T),
                verbose::Bool=false
        ) where T <: AbstractFloat
    AAG = AMHS.AAG
    CO = AAG.CO
    ff = vcat(CO.RR02*f, transpose(ig), transpose(og))
    ffh = fft(ff, 2)
    # here goes the gmres call
    NB = AAG.M*AAG.n
    F = FunctionOperator(
            (du,u,p,t)->annular_modified_helmholtz_apply(du, u, AMHS.AAG, RAG, AMHS.helmholtz_k, ia, ib, oa, ob),
            zeros(Complex{T}, NB), zeros(Complex{T}, NB)
        )
    P = MyPreconditioner(AMHS.KINVS, AAG.M, AAG.n)
    LP = LinearProblem(F, vec(ffh))
    out = solve(LP, KrylovJL_GMRES(), Pr=P, abstol=atol, reltol=rtol, verbose=verbose)
    uh = reshape(AMHS.AAG, out.u)
    return real.(ifft(uh, 2))
end

# need to reshape on the entrances into here!
function annular_modified_helmholtz_apply(uh, AAG, RAG, helmholtz_k, ia, ib, oa, ob)
    uh = reshape(AAG, uh)
    CO = AAG.CO
    luh = scalar_laplacian(AAG, RAG, uh)
    fuh = helmholtz_k^2 .* CO.RR02*uh - luh
    ibc = (@. ia*CO.IBCD + ib*CO.IBCN)*uh
    obc = (@. oa*CO.OBCD + ob*CO.OBCN)*uh
    return vec(vcat(fuh, ibc, obc))
end
function annular_modified_helmholtz_apply(duh, uh, AAG, RAG, helmholtz_k, ia, ib, oa, ob)
    out = annular_modified_helmholtz_apply(uh, AAG, RAG, helmholtz_k, ia, ib, oa, ob)
    duh .= out
    return duh
end

end
