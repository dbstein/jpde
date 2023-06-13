push!(LOAD_PATH, pwd())

using Revise
using PeriodicInterpolaters
using FFTW
using BenchmarkTools

# points in signal
Nsignal = 1000
# points to test
Ntest = 1000

# generate random points
t = LinRange(0.0, 2π, Nsignal+1)[1:Nsignal];
y = @. exp(cos(t));
# note the eval points can be anywhere, they just get modded by 2π
x = rand(Ntest).*100 .- 50;

# direct interpolation
ye = real.(DirectPeriodicInterpolate(x, fft(y)));
ya = @. exp(cos(x));
err = @. abs(ya - ye);
print("Error from direct interpolation is: ", maximum(err))

# time test the non-allocating version
out = zeros(Complex{Float64}, Ntest);
fh = fft(y);
@btime DirectPeriodicInterpolate!($out, $x, $fh);

# test the single-function periodic version
ε = 1e-12
PI = PeriodicInterpolater(y, ε);
ye2 = PI(x);
err = @. abs(ya - ye2);
print("Error from nufft interpolation is: ", maximum(err))

# again, time-test the non-allocating version
out = similar(x);
@btime $PI($out, $x);

# test complex version (interpolating complex function)
y = @. exp(im*cos(t));
PI = PeriodicInterpolater(y, ε);
ya = @. exp(im*cos(x));
ye2 = PI(x);
err = @. abs(ya - ye2);
print("Error from nufft interpolation is: ", maximum(err))

# again, time-test the non-allocating version
out = similar(ye2);
@btime $PI($out, $x);

# now test the batched version, for complex variables
yy = Array(transpose(hcat(exp.(im.*cos.(t)), exp.(im.*sin.(t)), sin.(t)+im.*sin.(t))));
PI = PeriodicInterpolater(yy, ε);
out = zeros(ComplexF64, 3, Ntest);
ye = PI(out, x);
ya = Array(transpose(hcat(exp.(im.*cos.(x)), exp.(im.*sin.(x)), sin.(x)+im.*sin.(x))));
err = @. abs(ya - ye);
print("Error from nufft interpolation is: ", maximum(err))
@btime $PI($out, $x);

# now test the batched version, for real variables
yy = Array(transpose(hcat(exp.(cos.(t)), exp.(sin.(t)), sin.(t).+cos.(t), sin.(t), cos.(t), exp.(cos.(t)).-exp.(sin.(t)))));
PI = PeriodicInterpolater(yy, ε);
out = zeros(Float64, 6, Ntest);
ye = PI(out, x);
ya = Array(transpose(hcat(exp.(cos.(x)), exp.(sin.(x)), sin.(x).+cos.(x), sin.(x), cos.(x), exp.(cos.(x)).-exp.(sin.(x)))));
err = @. abs(ya - ye);
print("Error from nufft interpolation is: ", maximum(err))
@btime $PI($out, $x);


