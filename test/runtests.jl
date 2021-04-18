using CUDA
using DelimitedFiles
using TabulatedFunctions
using Test


# ******************************************************************************
K = 7
foo(x) = 3 * x^7


# ******************************************************************************
N = 256

xmin, xmax = 10.0, 100.0
xlog10 = range(log10(xmin), log10(xmax), length=N)
x = @. 10^xlog10

y = @. foo(x)

tfname = "tmp.tf"

xu, yu = 1.0, 1.0

open(tfname, "w") do io
    DelimitedFiles.writedlm(io, [x y])
end

tf = TabulatedFunctions.TFunction(Float64, tfname, xu, yu)

tfgpu = TabulatedFunctions.TFunction(Float32, tfname, xu, yu)

rm(tfname)


# ******************************************************************************
# CPU
# ******************************************************************************
# arguments at the same points:
@test isapprox(tf.(x), foo.(x))

Ks = @. TabulatedFunctions.tfpower(tf, x)
@test isapprox(Ks, ones(N) * K)


# arguments at intermediate points and outside of the boundaries:
N1 = 10 * N
x1 = range(xmin - xmin/2, xmax + xmax/2, length=N1)
@test isapprox(tf.(x1), foo.(x1))

Ks1 = @. TabulatedFunctions.tfpower(tf, x1)
@test isapprox(Ks1, ones(N1) * K)


# ******************************************************************************
# GPU
# ******************************************************************************
function kernel_tf(y, tf, x)
    i = CUDA.threadIdx().x
    y[i] = tf(x[i])
    return nothing
end


function kernel_tfpower(Ks, tf, x)
    i = CUDA.threadIdx().x
    Ks[i] = TabulatedFunctions.tfpower(tf, x[i])
    return nothing
end


# arguments at the same points:
N = 256

xlog10 = range(log10(xmin), log10(xmax), length=N)
x = @. 10^xlog10
xgpu = CUDA.CuArray{Float32}(x)

ygpu = CUDA.zeros(Float32, N)
@cuda threads=N kernel_tf(ygpu, tfgpu, xgpu)
@test isapprox(collect(ygpu), collect(foo.(xgpu)))

Ksgpu = CUDA.zeros(Float32, N)
@cuda threads=N kernel_tfpower(Ksgpu, tfgpu, xgpu)
@test isapprox(collect(Ksgpu), ones(N) * K)


# arguments at intermediate points and outside of the boundaries:
N1 = 1024

x1 = range(xmin - xmin/2, xmax + xmax/2, length=N1)
x1gpu = CUDA.CuArray{Float32}(x1)

y1gpu = CUDA.zeros(Float32, N1)
@cuda threads=N1 kernel_tf(y1gpu, tfgpu, x1gpu)
@test isapprox(collect(y1gpu), collect(foo.(x1gpu)))

Ks1gpu = CUDA.zeros(Float32, N1)
@cuda threads=N1 kernel_tfpower(Ks1gpu, tfgpu, x1gpu)
@test isapprox(collect(Ks1gpu), ones(N1) * K)
