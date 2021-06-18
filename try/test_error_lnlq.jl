using Krylov, LinearAlgebra, MatrixMarket, Printf, SparseArrays, Test

# https://sparse.tamu.edu/
# 1st test done in LNLQ paper: scagr7-2c 	Meszaros https://sparse.tamu.edu/Meszaros/scagr7-2c
A = MatrixMarket.mmread("try/scagr7-2c/scagr7-2c.mtx")
# A = rand(2447, 3479)
@test size(A) == (2447, 3479)
σₐ = 2.422216e-02 # given on the website
σₑₛₜ = (1 - 1e-1) * σₐ
λ = 0.

#=
A = MatrixMarket.mmread("try/lp_kb2/lp_kb2.mtx")
@test size(A) == (43, 68)
σₐ = 1.233973e-02 # given on the website
σₑₛₜ = (1 - 1e-7) * σₐ
λ = 0. # 0.5 * σₐ
=#

(m, n) = size(A)
b = ones(m)/√m
itmax = 1000
# sol = A \ b # returns something else
sol = pinv(Matrix(A))* b
soly = pinv(Matrix(A * A')) * b

function xhist_to_sol(xhist, sol)
  N = length(xhist)
  res = Array{Float64}(undef, N)
  for i=1:N
    res[i] = norm(xhist[i] - sol)
  end
  return res
end

# include("lnlq.jl")
# include("craig.jl")

println("Test solve with CRAIG")
(x3, y3, xhist3, yChist3, stats3) = Krylov.craig_bis(A, b, history=true, itmax = itmax)
r3 = b - A * x3
resid3 = norm(r3) / norm(b)
@show resid3
stats3.residuals
xres3 = xhist_to_sol(xhist3, sol)
yCres3 = xhist_to_sol(yChist3, soly)

println("Test solve with LNLQ")
(x1, y1, stats1, xLhist1, xChist1, yLhist1, yChist1, errvec_x1, errvec_xL1, errvec_yC1, errvec_yL1) = Krylov.lnlq_bis(A, b, transfer_to_craig=false, history=true, σₐ = σₑₛₜ, itmax = itmax, λ = λ)
r1 = b - A * x1
resid1 = norm(r1) / norm(b)
@show resid1
stats1.residuals
xLres1 = xhist_to_sol(xLhist1, sol)
xCres1 = xhist_to_sol(xChist1, sol)
yCres1 = xhist_to_sol(yChist1, soly) # yChist1[end])
yLres1 = xhist_to_sol(yLhist1, soly) # yLhist1[end])

println("Test solve with CRAIG (from LNLQ transfer)")
(x2, y2, stats2, xLhist2, xChist2, yLhist2, yChist2, errvec_x2, errvec_xL2, errvec_yC2, errvec_yL2) = Krylov.lnlq_bis(A, b, transfer_to_craig=true, history=true, σₐ = σₑₛₜ, itmax = itmax, λ = λ)
r2 = b - A * x2
resid2 = norm(r2) / norm(b)
@show resid2
stats2.residuals
xLres2 = xhist_to_sol(xLhist2, sol)
xCres2 = xhist_to_sol(xChist2, sol)
yCres2 = xhist_to_sol(yChist2, soly) # yChist2[end])
yLres2 = xhist_to_sol(yLhist2, soly) # yLhist2[end])

using Plots
plot(log10.(xCres1), title="Error in x, log scale", label=["||xₗₙₗ-sol||"], legend=:bottomleft)
plot!(log10.(errvec_x1), label=["bound_xₗₙₗ"])
plot!(log10.(xCres2), label=["||xlnlqcraig-sol||"])
plot!(log10.(xres3), label=["||xcraig-sol||"])
png("xC-plot")

plot(log10.(xLres1), title="Error in x, log scale", label=["||xₗₙₗ-sol||"], legend=:bottomleft)
plot!(log10.(errvec_xL1), label=["bound_xₗₙₗ"]) # pas la bonne !!!!
plot!(log10.(xLres2), label=["||xlnlqcraig-sol||"])
plot!(log10.(xres3), label=["||xcraig-sol||"])
png("xL-plot")

plot(log10.(yCres1), title="Error in yC, log scale", label=["||yCₗₙₗ-sol||"], legend=:bottomleft)
plot!(log10.(errvec_yC1), label=["bound_yCₗₙₗ"])
plot!(log10.(yCres2), label=["||yClnlqcraig-sol||"])
plot!(log10.(yCres3), label=["||ycraig-sol||"])
png("yC-plot")

plot(log10.(yLres1), title="Error in yL, log scale", label=["||yLₗₙₗ-sol||"], legend=:bottomleft)
plot!(log10.(errvec_yL1), label=["bound_yLₗₙₗ"])
plot!(log10.(yLres2), label=["||yLlnlqcraig-sol||"])
plot!(log10.(yCres3), label=["||ycraig-sol||"])
png("yL-plot")