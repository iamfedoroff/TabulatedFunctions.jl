using DelimitedFiles
using TabulatedFunctions
using Test


# ******************************************************************************
sigma = 3
K = 7
f(x) = sigma * x^K

tfname = "tmp.tf"

xu, yu = 1.0, 1.0

xmin, xmax, N = 10.0, 100.0, 10
xlog10 = range(log10(xmin), log10(xmax), length=N)

x = @. 10^xlog10
y = @. f(x)


# ------------------------------------------------------------------------------
open(tfname, "w") do io
    DelimitedFiles.writedlm(io, [x y])
end

tf = TabulatedFunctions.TFunction(Float64, tfname, xu, yu)

rm(tfname)


# ******************************************************************************
# arguments at the same points:
@test isapprox(tf.(x), f.(x))

Ks = ones(N) * K
Kstf = @. TabulatedFunctions.tfpower(tf, x)
@test isapprox(Kstf, Ks)

# arguments at intermediate points and outside of the boundaries:
N1 = 10 * N
x1 = range(xmin - xmin/2, xmax + xmax/2, length=N1)
@test isapprox(tf.(x1), f.(x1))

Ks1 = ones(N1) * K
Kstf1 = @. TabulatedFunctions.tfpower(tf, x1)
@test isapprox(Kstf1, Ks1)
