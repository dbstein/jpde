module NearFinding

using FourierTools
using PeriodicInterpolaters
using Threaded

export gridpoints_near_points, gridpoints_near_points!, gridpoints_near_curve

"""
Fast near-points finder for a grid and set of points. 

When bx/by describe a polygon, one may use this function to find all points
within a distance D of the polygon, by setting:
d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
segment.  If l < D, then d need only be 1.12D to guarantee all near-points
are found.  Note that points that are not within D of the polygon will aslo
be marked as "near", however

The function computes:
    three arrays with size [nx, ny]; n* = length(*v)
    1) Bool(nx, ny)  --> are you within d of the boundary?
    2) Int64(nx, ny) --> two which boundary point are you closest?
    3) T(nx, ny)     --> how far are you from the closest point?

Inputs (for T an AbstractFloat):
    bx, T(nb): x-coordinates of boundary
    by, T(nb): y-coordinates of boundary
    xv, T(nx): x-values for grid coordinates
    yv: T(ny): y-values for grid coordinates
    d:  distance to find near points
Outputs:
    is_close,     Bool(nx, ny),  is this point within d of any boundary point?
    closest_ind,  Int64(nx, ny), index of closest boundary point to this point
    closest_d2,   T(nx, ny),     closest squared distance to a boundary point
"""
function gridpoints_near_points(bx::AbstractVector{T}, by::AbstractVector{T}, xv::AbstractVector{T}, yv::AbstractVector{T}, d::T) where T <: AbstractFloat
    Nx = length(xv)
    Ny = length(yv)

    # allocate storage arrays
    is_close    = zeros(Bool,     Nx, Ny)
    closest_ind = fill(Int64(-1), Nx, Ny)
    closest_d2  = fill(T(1e15),   Nx, Ny)

    gridpoints_near_points!(is_close, closest_ind, closest_d2, bx, by, xv, yv, d)
    return is_close, closest_ind, closest_d2
end

function gridpoints_near_points!(is_close::AbstractMatrix{Bool}, closest_ind::AbstractMatrix{Int64}, closest_d2::AbstractMatrix{T}, bx::AbstractVector{T}, by::AbstractVector{T}, xv::AbstractVector{T}, yv::AbstractVector{T}, d::T) where T <: AbstractFloat
    Nx = length(xv)
    Ny = length(yv)

    # number of points we're searching over
    Nb = length(bx)
    # regular grid spacing
    xh = xv[begin+1] - xv[begin]
    yh = yv[begin+1] - yv[begin]
    # search distances
    xsd = ceil(Int64, d/xh)
    ysd = ceil(Int64, d/yh)
    d2 = d*d
    # lower bounds
    xlb = xv[begin]
    ylb = yv[begin]
    @inbounds(
    for i in eachindex(bx, by)
        # locate indeces to lower left of the point
        x_loc = floor(Int64, (bx[i] - xlb) / xh) + firstindex(xv)
        y_loc = floor(Int64, (by[i] - ylb) / yh) + firstindex(yv)
        # get indeces in x/ y to search over
        x_lower = max(x_loc - xsd, firstindex(xv))
        x_upper = min(x_loc + xsd, lastindex(xv))
        y_lower = max(y_loc - ysd, firstindex(yv))
        y_upper = min(y_loc + ysd, lastindex(yv))
        # loop over the square
        for j in x_lower:x_upper
            for k in y_lower:y_upper
                # compute x dist/y dist/squared dist
                xd = xv[j] - bx[i]
                yd = yv[k] - by[i]
                dist2 = xd^2 + yd^2
                # if we're close, mark that we are
                is_close[j, k] = is_close[j, k] || (dist2 < d2)
                # if we're closest so far, mark that we are
                closer = is_close[j, k] && (dist2 < closest_d2[j, k])
                closest_d2[j, k] = closer ? dist2 : closest_d2[j, k]
                closest_ind[j, k] = closer ? i : closest_ind[j, k]
            end
        end
    end
    )
    @. closest_d2 = ifelse(closest_d2 > T(1e14), T(NaN), closest_d2)
    # closest[closest .> 1e14] .= NaN
    return is_close, closest_ind, closest_d2
end

"""
Computes the speed of a closed plane curve (x(s), y(s))
given points x(s_i), y(s_i)
the final point is assumed to be not repeated
"""
function compute_speed(x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat
    n = length(x)
    ik = Complex{T}(im)*rfftfreq(n, T(n))
    xp = irfft(ik.*rfft(x), n)
    yp = irfft(ik.*rfft(y), n)
    return hypot.(xp, yp)
end

"""
Upsamples the curve (cx, cy) so that it can be used in nearest curve finding
In particular, we insist that the largest point-point distance is <= d
Inputs:
    cx, float(n): x-coordinates of curve
    cy, float(n): y-coordinates of curve
    d,  float:    search distance
Returns:
    cx, float(N): x-coordinates of upsampled curve
    cy, float(N): y-coordinates of upsampled curve
    D,  float:    search distance to be used in near-points finder
"""
function upsample_curve(cx::AbstractVector{T}, cy::AbstractVector{T}, d::T) where T <: AbstractFloat
    # compute the speed of the approximation
    n = length(cx)
    dt = 2T(π)/n
    speed = compute_speed(cx, cy)
    max_h = maximum(speed)*dt
    # if the curve is too poorly resolved to compute things accurate, upsample
    if max_h > d
        n *= ceil(Int64, max_h/d)
        cx = resample(cx, n)
        cy = resample(cy, n)
    end
    # extra large fudge factor for near search because its a curve
    D = T(1.5)*d
    return cx, cy, D
end

"""
Computes, for all gridpoints spanned by (xv, yv), whether the gridpoints
1) are within d of the (closed) curve
2) for those that are within d of the curve,
    how far the gridpoints are from the curve
    (that is, closest approach in the normal direction)
3) local coordinates, in (r, t) coords, for the curve
    that is, we assume the curve is given by X(t) = (cx(t_i), cy(t_i))
    we define local coordinates X(t, r) = X(t) + n(t) r, where n is
    the normal to the curve at t,
    and return (r, t) for each gridpoint in some search region

Inputs:
    cx,                   float(nb): x-coordinates of boundary
    cy,                   float(nb): y-coordinates of boundary
    xv,                   float(nx): x-values for grid coordinates
    yv,                   float(ny): y-values for grid coordinates
    d,                    float:     distance to search within 
Outputs: (tuple of)
    in_annulus, bool (nx, ny), whether points are in annulus of radius d
    r,          float(nx, ny), r-coordinate for points in_annulus
    t,          float(nx, ny), t-coordinate for points in_annulus
    (d, cx, cy) float,         search distance, upsampled cx, cy
"""
function gridpoints_near_curve(cx::AbstractVector{T}, cy::AbstractVector{T}, xv::AbstractVector{T}, yv::AbstractVector{T}, d::T; ε::T=100eps(T)) where T <: AbstractFloat
    # upsample if needed and get search distance
    cx, cy, D = upsample_curve(cx, cy, d)
    # get points near points
    near, guess_ind, _ = gridpoints_near_points(cx, cy, xv, yv, D)
    # allocate output arrays
    Nx = length(xv)
    Ny = length(yv)
    # allocate storage arrays
    in_annulus = zeros(Bool, Nx, Ny)
    r = fill(T(NaN), Nx, Ny)
    θ = fill(T(NaN), Nx, Ny)
    # if there are any near points, compute local coordinates
    inds = findall(near)
    if length(inds) > 0
        LCP = LocalCoordinatePrecompute(cx, cy)
        @threaded for ind in inds
            @inbounds gt = 2T(π)*(guess_ind[ind]-1)/length(cx)
            @inbounds θh, rh = compute_local_coordinates(xv[ind[1]], yv[ind[2]], LCP, gt, ε=ε)
            if abs(rh) <= d
                @inbounds in_annulus[ind] = true
                @inbounds r[ind] = rh
                @inbounds θ[ind] = mod(θh, 2T(π))
            end
        end
    end
    return in_annulus, r, θ
end

struct LocalCoordinatePrecompute{T}
    I::PeriodicInterpolater{T, T, 6}
end
function LocalCoordinatePrecompute(cx::AbstractVector{T}, cy::AbstractVector{T}; ε::T=100eps(T)) where T <: AbstractFloat
    n = length(cx)
    # fourier stuff
    dt = 2T(π)/n
    ts = Vector(1:n-1)*dt
    ik = Complex{T}(im)*rfftfreq(n, T(n))
    xh = rfft(cx)
    yh = rfft(cy)
    # derivatives
    xp = irfft(ik.*xh, n)
    yp = irfft(ik.*yh, n)
    xpp = irfft(ik.^2 .* xh, n)
    ypp = irfft(ik.^2 .* yh, n)
    # interpolation operators
    A = Array(transpose(hcat(cx, cy, xp, yp, xpp, ypp)));
    I = PeriodicInterpolater(A; ε=ε)
    return LocalCoordinatePrecompute{T}(I)
end

@inline function coord_interp(t::T, LCP::LocalCoordinatePrecompute{T}) where T <: AbstractFloat
    out = LCP.I(t)
    X = out[1]
    Y = out[2]
    Xp = out[3]
    Yp = out[4]
    Xpp = out[5]
    Ypp = out[6]
    return X, Y, Xp, Yp, Xpp, Ypp
end
@inline function coord_obj_jac(t::T, x::T, y::T, LCP::LocalCoordinatePrecompute{T}) where T <: AbstractFloat
    X, Y, Xp, Yp, Xpp, Ypp = coord_interp(t, LCP)
    obj = Xp*(X-x) + Yp*(Y-y)
    jac = Xpp*(X-x) + Ypp*(Y-y) + Xp*Xp + Yp*Yp
    return obj, jac
end
"""
Dirt simple non-allocating Newton for coordinate solves
Makes use of the fact that I can evalute the objective essentially for free
    after the Jacobian has been evaluated
"""
function SimpleNewton(fj, x::T, ε::T) where T <: AbstractFloat
    r, J = fj(x)
    it = 1
    while abs(r) > ε
        if it > 30
            error("Newton solve for coordinates took too many iterations.")
        end
        x -= r/J
        r, J = fj(x)
        it += 1
    end
    return x
end

"""
Find using the coordinates:
x = X + r n_x
y = Y + r n_y
"""
function compute_local_coordinates(x::T, y::T, LCP::LocalCoordinatePrecompute{T}, guess_t::T; ε::T=100eps(T)) where T <: AbstractFloat
    t = SimpleNewton(t -> coord_obj_jac(t, x, y, LCP), guess_t, ε)
    X, Y, Xp, Yp, Xpp, Ypp = coord_interp(t, LCP)
    # compute r
    r = hypot(X-x, Y-y)
    # compute unit normal vectors
    isp = 1/hypot(Xp, Yp)
    nx = Yp * isp
    ny = -Xp * isp
    # determine sign
    d1 = hypot(X + nx*r - x, Y + ny*r - y)
    d2 = hypot(X - nx*r - x, Y - ny*r - y)
    sign = d1 < d2 ? 1 : -1
    return t, r*sign
end

end
