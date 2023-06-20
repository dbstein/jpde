module Kernels

export ModifiedHelmholtzSLP#, ModifiedHelmholtzDLP, ModifiedHelmholtzKernel

using SpecialFunctions
using Threaded

@inline function _ModifiedHelmholtzKernel(
        source::StaticVector{2, T},
        target::StaticVector{2, T},
        force::T,
        dipole::T
    ) where T <: AbstractFloat
    dx = source[1] - target[1]
    dy = source[2] - target[2]
    r = hypot(dx, dy)
    return k0(k*r)
end
function ModifiedHelmholtzSLP(
                    source::AbstractMatrix{T}, 
                    force::AbstractVector{T},
                    target::AbstractMatrix{T},
                )
    dipole = zero(force)
    potential = zero(force)
    return ModifiedHelmholtzKernel!(potential, source, force, dipole, target)
end
function ModifiedHelmholtzKernel!(
        potential::AbstractVector{T},
        source::AbstractMatrix{T}, 
        force::AbstractVector{T},
        dipole::AbstractVector{T},
        target::AbstractMatrix{T},
    ) where T <: AbstractFloat
    return GenericScalarKernel!(potential, source, force, dipole, target, _ModifiedHelmholtzKernel)
end
function ModifiedHelmholtzKernelSelf!(
        potential::AbstractVector{T},
        source::AbstractMatrix{T}, 
        force::AbstractVector{T},
        dipole::AbstractVector{T},
        target::AbstractMatrix{T},
    ) where T <: AbstractFloat
    return GenericScalarKernelSelf!(potential, source, force, dipole, target, _ModifiedHelmholtzKernel)
end

function GenericScalarKernel!(
        potential::AbstractVector{T},
        source::AbstractMatrix{T}, 
        force::AbstractVector{T},
        dipole::AbstractVector{T},
        target::AbstractMatrix{T},
        kernel::Function
    )
    potential .= zero(T)
    sourcer = reinterpret(SVector{2, T}, source)
    targetr = reinterpret(SVector{2, T}, target)
    @threaded for j in eachrow(target)
        for i in eachrow(source)
            potential[j] += kernel(source[i], target[j], force[i], dipole[i])
        end
    end
    return potential
end
function GenericScalarKernelSelf!(
        potential::AbstractVector{T},
        source::AbstractMatrix{T}, 
        force::AbstractVector{T},
        dipole::AbstractVector{T},
        target::AbstractMatrix{T},
        kernel::Function
    )
    potential .= zero(T)
    sourcer = reinterpret(SVector{2, T}, source)
    targetr = reinterpret(SVector{2, T}, target)
    @threaded for j in eachrow(target)
        for i in eachrow(source)
            potential[j] += (i==j) ? zero(T) : kernel(source[i], target[j], force[i], dipole[i])
        end
    end
    return potential
end

end