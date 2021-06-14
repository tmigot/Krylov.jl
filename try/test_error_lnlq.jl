using Krylov, LinearAlgebra, MatrixMarket, Printf, SparseArrays, Test

# https://sparse.tamu.edu/
# 1st test done in LNLQ paper: scagr7-2c 	Meszaros https://sparse.tamu.edu/Meszaros/scagr7-2c
#=
A1 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_A_1.mtx")
A2 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_A_2.mtx")
A3 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_A_3.mtx")

b1 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_b_1.mtx")
b2 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_b_2.mtx")
b3 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_b_3.mtx")

c1 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_c_1.mtx")
c2 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_c_2.mtx")
c3 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_c_3.mtx")

hi1 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_hi_1.mtx")
hi2 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_hi_2.mtx")
hi3 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_hi_3.mtx")

lo1 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_lo_1.mtx")
lo2 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_lo_2.mtx")
lo3 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_lo_3.mtx")

c1 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_z0_1.mtx")
c2 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_z0_2.mtx")
c3 = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c_z0_3.mtx")
=#
A = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c.mtx")
@test size(A) == (2447,3479)
(m, n) = size(A)
b = ones(m)/√m
# sol = A \ b # returns something else
sol = pinv(Matrix(A))* b

σₐ = 2.422216e-02 # given on the website
σₑₛₜ = (1 - 1e-10) * σₐ

function xhist_to_sol(xhist, sol)
  N = length(xhist)
  res = Array{Float64}(undef, N)
  for i=1:N
    res[i] = norm(xhist[i] - sol)
  end
  return res
end

include("lnlq.jl")
println("Test solve with LNLQ")
(x1, y1, stats1, xhist1, yLhist1, yChist1, errvec_x1, errvec_y1) = lnlq(A, b, transfer_to_craig=false, history=true, σₐ = σₐ)
r1 = b - A * x1
resid1 = norm(r1) / norm(b)
@show resid1
stats1.residuals
xres1 = xhist_to_sol(xhist1, sol)
yres1 = xhist_to_sol(yChist1, y1)

println("Test solve with CRAIG (from LNLQ transfer)")
(x2, y2, stats2, xhist2, yLhist2, yChist2, errvec_x2, errvec_y2) = lnlq(A, b, transfer_to_craig=true, history=true, σₐ = σₐ)
r2 = b - A * x2
resid2 = norm(r2) / norm(b)
@show resid2
stats2.residuals
xres2 = xhist_to_sol(xhist2, sol)
yres2 = xhist_to_sol(yChist2, y2)

using Plots
plot(log.(xres1), title="Error in x, log scale", label=["||xₗₙₗ-sol||"])
plot!(log.(errvec_x1), label=["bound_xₗₙₗ"])
plot!(log.(xres2), label=["||xcraig-sol||"])
png("x-plot")

plot(log.(xres1), title="Error in x, log scale", label=["||xₗₙₗ-sol||"])
plot!(log.(errvec_x1), label=["bound_xₗₙₗ"])
plot!(log.(xres2), label=["||xcraig-sol||"])
png("x-plot")