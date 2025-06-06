using Test
using Pkg
using Random
using Revise
Pkg.activate(".")

includet("../src/BNPStep.jl")
using .BNPStep 

@testset "BNP-Step Self-Consistency Test" begin
    # Simulate a toy step function (10 steps, 1000 points)
    N = 1000
    t = collect(1:N)
    true_steps = repeat(1:10, inner=N÷10)
    noise = 0.1f0 .* randn(Float32, N)
    data = true_steps .+ noise
    dataset = Dict("data" => data, "times" => Float32.(t))

    # Run BNP-Step with minimal iterations
    step_model = BNP_Step(gamma=1.0f0, B_max=20)
    results = analyze(step_model, dataset, 100)

    # Save and reload results using actual HDF5 I/O
    tmpfile = tempname()
    save_results(tmpfile, results)
    @info "Saved results to $tmpfile"
    reloaded = load_results(tmpfile)

    # === Consistency checks ===
    @test haskey(reloaded, "f_bg")
    @test haskey(reloaded, "h_m")
    @test haskey(reloaded, "b_m")
    @test length(reloaded["f_bg"]) == 100
    @test !any(isnan, reloaded["f_bg"])
    @test !any(isnan, reduce(vcat, reloaded["b_m"]))
end
