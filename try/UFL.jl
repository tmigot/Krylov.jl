using LinearAlgebra, SparseArrays
using LinearOperators, Printf
using MatrixMarket
using Krylov

#=
include("utils.jl")
include("craig.jl")
include("lnlq.jl")
include("symmlq.jl")
include("minres.jl")
include("blcg_blminres.jl")
=# 

# under-determined systems
ud = ["lp_osa_07", "lp_czprob", "lp_d6cube", "lpi_klein3", "lp_80bau3b", "lp_adlittle", "lp_blend", "lp_etamacro"]

# over-determined problems
od = ["well1033", "well1850", "illc1850", "landmark", "Maragal_6", "tomographic1"]

dict = Dict("lp_osa_07"    => false,
            "lp_czprob"    => true,
            "lp_d6cube"    => false,
            "lpi_klein3"   => false,
            "well1033"     => false,
            "well1850"     => false,
            "illc1850"     => false,
            "landmark"     => false,
            "Maragal_6"    => false,
            "tomographic1" => false,
            "lp_80bau3b"   => false,
            "lp_adlittle"  => false,
            "lp_blend"     => false,
            "lp_etamacro"  => false)

iterations = Dict("lp_osa_07"    => 400,
                  "lp_czprob"    => 175,
                  "lp_d6cube"    => 450,
                  "lpi_klein3"   => 3500,
                  "well1033"     => 45,
                  "well1850"     => 60,
                  "illc1850"     => 70,
                  "landmark"     => 125,
                  "Maragal_6"    => 310,
                  "tomographic1" => 200,
                  "lp_80bau3b"   => 500,
                  "lp_adlittle"  => 300,
                  "lp_blend"     => 300,
                  "lp_etamacro"  => 600)

atol = 1e-12
rtol = 1e-10

using Plots

for pb in keys(dict)

  if dict[pb]
    println("Problème $pb")
    nb_iter = iterations[pb]
  
    println("Lecture des données: ")
    A = Float64.(MatrixMarket.mmread("./try/UFL/" * pb * ".mtx"))
    m, n = size(A)
    println("Dimension de A: ($m, $n)")
    println("Nombre de nnz: ", length(A.nzval))
    b = ones(m)
    # b  = A * sol
    sol = pinv(Matrix(A))* b

    #=
    A = MatrixMarket.mmread("try/lp_kb2/lp_kb2.mtx")
    σₐ = 1.233973e-02 # given on the website
    (m, n) = size(A)
    b = ones(m)/√m
    itmax = 1000
    # sol = A \ b # returns something else
    sol = pinv(Matrix(A))* b
    =#
    println("✔")

    r0 = norm(b)
    res_craig    = [r0; zeros(nb_iter)]
    res_lnlq    = [r0; zeros(nb_iter)]

    println("Résolution du système linéaire: ")
    for itmax = 1 : nb_iter
      GC.gc()
      println(itmax)
      x_craig, y_craig, stats_craig = craig(A, b, atol=0.0, rtol=0.0, btol=0.0, conlim=Inf, itmax=itmax+1, history=true)
      x_lnlq, y_lnlq, stats_lnlq = lnlq(A, b, atol=0.0, rtol=0.0, itmax=itmax, history=true)

      res_craig[itmax+1]    = norm(sol - x_craig)
      res_lnlq[itmax+1]    = norm(sol - x_lnlq)
      @show norm(x_craig-x_lnlq), length(stats_lnlq.residuals), length(stats_craig.residuals)
    end
    println("✔")

    pos_craig    = length(res_craig) # analyse(res_craig   , atol, rtol, r0)
    pos_lnlq     = length(res_lnlq) # analyse(res_lnlq   , atol, rtol, r0)

    print("Génération des graphiques: ")
    plot(0:pos_lnlq-1   , res_lnlq[1:pos_lnlq]    , label="lnlq"      , yaxis=:log10, lw=1, color=1, line=:dash)
    plot!(0:pos_craig-1 , res_craig[1:pos_craig]  , label="craig"   , yaxis=:log10, lw=1, color=2, legend=:best)
    png("$(pb)_craig_lnlq")
    # savefig("$(pb)_craig_lnlq.tex")
    # run(`mv graphiques/$(pb)_craig_lnlq.tex graphiques/$(pb)_craig_lnlq.tikz`)
    # run(`./sed1.sh $(pb)_craig_lnlq`)
    println("✔")
  end
end
