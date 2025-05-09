module BNPStep

"""
Main module for BNP-Step.

This module serves as the entry point for running BNP-Step analysis.
"""

include("BNPAnalysis.jl")
using ..BNPAnalysis
include("BNPInputs.jl")
using ..BNPInputs
include("BNPSampler.jl")
using ..BNPSampler


using Random
using LinearAlgebra
using DelimitedFiles  # Required for reading files
using StatsBase  # For statistical operations
using Plots      # For visualization
using Distributions  # For sampling distributions


"""
    struct BNPStep

Defines the BNPStep structure for running BNP-Step analysis.

# Fields
- `chi::Float64`: Precision for Normal priors on h_m.
- `dt_ref::Float64`: Reference duration for Gamma prior on step durations.
- `h_ref::Float64`: Mean for Normal priors on h_m.
- `psi::Float64`: Precision for Normal prior on F_bg.
- `F_ref::Float64`: Mean for Normal prior on F_bg.
- `phi::Float64`: Shape parameter for Gamma prior on eta.
- `eta_ref::Float64`: Scale parameter for Gamma prior on eta.
- `gamma::Float64`: Hyperparameter for Bernoulli priors on b_m.
- `B_max::Int`: Maximum possible number of steps.
- `load_initialization::String`: Initialization strategy for b_m.
- `use_annealing::Bool`: Whether to use simulated annealing.
- `init_temperature::Int`: Initial temperature for simulated annealing.
- `scale_factor::Float64`: Controls how fast the temperature drops off.
- `rng::Random.AbstractRNG`: Random number generator.
"""
struct BNP_Step
    chi::Float64
    dt_ref::Float64
    h_ref::Float64
    psi::Float64
    F_ref::Float64
    phi::Float64
    eta_ref::Float64
    gamma::Float64
    B_max::Int
    load_initialization::String
    use_annealing::Bool
    init_temperature::Int
    scale_factor::Float64
    rng::Random.AbstractRNG
end

"""
    BNPStep(; kwargs...)

Constructor for the BNPStep structure with default values.

# Keyword Arguments
- `chi::Float64`: Default is 0.028.
- `dt_ref::Float64`: Default is 100.0.
- `h_ref::Float64`: Default is 10.0.
- `psi::Float64`: Default is 0.0028.
- `F_ref::Float64`: Default is 0.0.
- `phi::Float64`: Default is 1.0.
- `eta_ref::Float64`: Default is 10.0.
- `gamma::Float64`: Default is 1.0.
- `B_max::Int`: Default is 50.
- `load_initialization::String`: Default is "prior".
- `use_annealing::Bool`: Default is false.
- `init_temperature::Int`: Default is 2250.
- `scale_factor::Float64`: Default is 1.25.
- `seed::Union{Int, Nothing}`: Default is nothing.
"""
function BNP_Step(; chi=0.028, dt_ref=100.0, h_ref=10.0, psi=0.0028, F_ref=0.0, phi=1.0, eta_ref=10.0,
                  gamma=1.0, B_max=50, load_initialization="prior", use_annealing=false,
                  init_temperature=2250, scale_factor=1.25, seed=nothing)
    rng = isnothing(seed) ? Random.GLOBAL_RNG : MersenneTwister(seed)
    return BNPStep(chi, dt_ref, h_ref, psi, F_ref, phi, eta_ref, gamma, B_max,
                   load_initialization, use_annealing, init_temperature, scale_factor, rng)
end

"""
    analyze(step::BNPStep, data::Dict, num_samples::Int=50000)

Analyzes a dataset using BNP-Step and stores the results.

# Arguments
- `data::Dict`: Dictionary containing the dataset with keys "data" and "times".
- `num_samples::Int`: Number of samples to generate. Default is 50000.
"""
function analyze(step::BNPStep, data::Dict, num_samples::Int=50000)
    # Validate input data
    if !haskey(data, "data") || !haskey(data, "times")
        error("Dataset must contain 'data' and 'times' keys.")
    end

    # Initialize variables
    data_points = data["data"]
    data_times = data["times"]
    num_data = length(data_points)
    t_n = data_times !== nothing ? data_times : collect(1:num_data)

    # Initialize samples
    b_m = falses(step.B_max)
    h_m = rand(Normal(step.h_ref, sqrt(1 / step.chi)), step.B_max)
    t_m = rand(t_n, step.B_max)
    f_bg = rand(Normal(step.F_ref, sqrt(1 / step.psi)))
    eta = rand(Gamma(step.eta_ref / step.phi, step.phi))
    dt = rand(Gamma(step.dt_ref / step.chi, step.chi))

    # Placeholder for results
    results = Dict(
        "b_m" => [],
        "h_m" => [],
        "t_m" => [],
        "f_bg" => [],
        "eta" => [],
        "dt" => [],
        "posterior" => []
    )

    # Gibbs sampling loop
    for _ in 1:num_samples
        # Sample F_bg and h_m
        f_bg, h_m = sample_fh(step.B_max, num_data, data_points, t_n, b_m, t_m, dt, eta, step.psi, step.chi, step.F_ref, step.h_ref, step.rng)

        # Sample b_m
        b_m = sample_b(step.B_max, num_data, data_points, t_n, b_m, h_m, t_m, dt, f_bg, eta, step.gamma, step.rng)

        # Sample t_m
        t_m = sample_t(step.B_max, num_data, data_points, t_n, b_m, h_m, t_m, dt, f_bg, eta, step.rng)

        # Sample eta
        eta = sample_eta(step.B_max, num_data, data_points, t_n, b_m, h_m, t_m, dt, f_bg, step.phi, step.eta_ref, step.rng)

        # Sample dt
        dt = sample_dt(step.B_max, num_data, data_points, t_n, b_m, h_m, t_m, dt, f_bg, step.chi, step.dt_ref, step.rng)

        # Calculate posterior
        posterior = calculate_logposterior(step.B_max, num_data, data_points, t_n, b_m, h_m, t_m, dt, f_bg, eta, step.chi, step.h_ref, step.gamma, step.phi, step.eta_ref, step.psi, step.dt_ref, step.F_ref)

        # Append results
        push!(results["b_m"], b_m)
        push!(results["h_m"], h_m)
        push!(results["t_m"], t_m)
        push!(results["f_bg"], f_bg)
        push!(results["eta"], eta)
        push!(results["dt"], dt)
        push!(results["posterior"], posterior)
    end

    return results
end

"""
    visualize_results(results::Dict, data::Dict; plot_type::String="step", font_size::Int=16, datacolor::Symbol=:gray, learncolor::Symbol=:orange)

Visualizes the results of BNP-Step analysis.

# Arguments
- `results::Dict`: Dictionary containing the results of the analysis.
- `data::Dict`: Dictionary containing the dataset with keys "data" and "times".
- `plot_type::String`: Type of plot to generate. Default is "step".
- `font_size::Int`: Font size for plot labels. Default is 16.
- `datacolor::Symbol`: Color for the data plot. Default is :gray.
- `learncolor::Symbol`: Color for the learned steps plot. Default is :orange.
"""
function visualize_results(results::Dict, data::Dict; plot_type::String="step", font_size::Int=16, datacolor::Symbol=:gray, learncolor::Symbol=:orange)
    # Extract data
    data_points = data["data"]
    data_times = data["times"]
    t_n = data_times !== nothing ? data_times : collect(1:length(data_points))

    # Extract results
    b_m = results["b_m"]
    h_m = results["h_m"]
    t_m = results["t_m"]
    f_bg = results["f_bg"]
    eta = results["eta"]

    if plot_type == "step"
        println("Generating step plot...")

        # Generate step plot data
        step_data = zeros(length(t_n))
        for i in 1:length(b_m)
            if b_m[i]
                step_start = findfirst(x -> x >= t_m[i], t_n)
                step_end = findfirst(x -> x >= t_m[i] + h_m[i], t_n)
                step_data[step_start:step_end] .= f_bg + h_m[i]
            end
        end

        # Plot data and steps
        plot(t_n, data_points, label="Data", color=datacolor, lw=1.5)
        plot!(t_n, step_data, label="Steps", color=learncolor, lw=2.0)
        xlabel!("Time")
        ylabel!("Signal")
        title!("BNP-Step Results")
        legend()
    elseif plot_type == "hist_step_height"
        println("Generating histogram of step heights...")

        # Flatten step heights for histogram
        step_heights = vcat(h_m...)
        histogram(step_heights, bins=20, label="Step Heights", color=learncolor, alpha=0.7)
        xlabel!("Step Height")
        ylabel!("Frequency")
        title!("Histogram of Step Heights")
        legend()
    elseif plot_type == "hist_dwell_time"
        println("Generating histogram of dwell times...")

        # Compute dwell times
        dwell_times = []
        for i in 1:length(b_m)
            active_indices = findall(b_m[i])
            append!(dwell_times, diff(vcat(0.0, t_m[i][active_indices], maximum(t_n))))
        end

        # Plot histogram
        histogram(dwell_times, bins=20, label="Dwell Times", color=learncolor, alpha=0.7)
        xlabel!("Dwell Time")
        ylabel!("Frequency")
        title!("Histogram of Dwell Times")
        legend()
    elseif plot_type == "hist_emission"
        println("Generating histogram of emission levels...")

        # Flatten emission levels for histogram
        emission_levels = vcat(f_bg .+ h_m...)
        histogram(emission_levels, bins=20, label="Emission Levels", color=learncolor, alpha=0.7)
        xlabel!("Emission Level")
        ylabel!("Frequency")
        title!("Histogram of Emission Levels")
        legend()
    elseif plot_type == "hist_height_separated"
        println("Generating separated histogram of step heights...")

        # Separate step heights by active Bernoulli variables
        for i in 1:length(b_m)
            active_heights = h_m[i][b_m[i]]
            histogram(active_heights, bins=10, label="Step Heights (Sample $i)", alpha=0.5, lw=1.5)
        end
        xlabel!("Step Height")
        ylabel!("Frequency")
        title!("Separated Histogram of Step Heights")
        legend()
    elseif plot_type == "hist_f"
        println("Generating histogram of background fluorescence (F_bg)...")

        # Plot histogram of F_bg
        histogram(f_bg, bins=20, label="F_bg", color=learncolor, alpha=0.7)
        xlabel!("Background Fluorescence (F_bg)")
        ylabel!("Frequency")
        title!("Histogram of Background Fluorescence")
        legend()
    elseif plot_type == "hist_eta"
        println("Generating histogram of noise parameters (eta)...")

        # Plot histogram of eta
        histogram(eta, bins=20, label="Eta", color=learncolor, alpha=0.7)
        xlabel!("Noise Parameter (Eta)")
        ylabel!("Frequency")
        title!("Histogram of Noise Parameters")
        legend()
    elseif plot_type == "survivorship"
        println("Generating survivorship plot...")

        # Compute dwell times
        dwell_times = []
        for i in 1:length(b_m)
            active_indices = findall(b_m[i])
            append!(dwell_times, diff(vcat(0.0, t_m[i][active_indices], maximum(t_n))))
        end

        # Sort dwell times and compute survivorship
        sorted_dwell_times = sort(dwell_times)
        survivorship = 1 .- cumsum(fill(1.0 / length(sorted_dwell_times), length(sorted_dwell_times)))

        # Plot survivorship curve
        plot(sorted_dwell_times, survivorship, label="Survivorship", color=learncolor, lw=2.0)
        xlabel!("Dwell Time")
        ylabel!("Survivorship")
        title!("Survivorship Plot")
        legend()
    elseif plot_type == "hist_dwell_separated"
        println("Generating separated histogram of dwell times...")

        # Separate dwell times by active Bernoulli variables
        for i in 1:length(b_m)
            active_indices = findall(b_m[i])
            dwell_times = diff(vcat(0.0, t_m[i][active_indices], maximum(t_n)))
            histogram(dwell_times, bins=10, label="Dwell Times (Sample $i)", alpha=0.5, lw=1.5)
        end
        xlabel!("Dwell Time")
        ylabel!("Frequency")
        title!("Separated Histogram of Dwell Times")
        legend()
    elseif plot_type == "hist_emission_separated"
        println("Generating separated histogram of emission levels...")

        # Separate emission levels by active Bernoulli variables
        for i in 1:length(b_m)
            active_emissions = f_bg + h_m[i][b_m[i]]
            histogram(active_emissions, bins=10, label="Emission Levels (Sample $i)", alpha=0.5, lw=1.5)
        end
        xlabel!("Emission Level")
        ylabel!("Frequency")
        title!("Separated Histogram of Emission Levels")
        legend()
    else
        println("Unsupported plot type: $plot_type")
    end
end

"""
    load_data(filename::String; kwargs...) -> Dict

Loads a dataset for BNP-Step analysis.
"""
function load_data(filename::String; kwargs...)
    # Implement logic to load data based on file type and options
    return Dict("data" => rand(100), "times" => collect(1:100))  # Example placeholder
end

"""
    save_results(filename::String, results::Dict)

Saves the results to a file.

# Arguments
- `filename::String`: Name of the file to save results.
- `results::Dict{String, Any}`: Dictionary containing results to save.
"""
function save_results(filename::String, results::Dict{String, Any})
    # Replace with actual file saving logic
    println("Saving results to ", filename)
end

"""
    main()

Main function for running BNP-Step analysis.
"""
function main()
    println("Running BNP-Step analysis...")

    # Load data
    data = load_data("example_data.txt")

    # Initialize BNPStep object
    step = BNPStep(seed=42)

    # Perform analysis
    results = analyze(step, data, num_samples=5000)

    # Visualize results
    visualize_results(results, data)

    println("BNP-Step analysis completed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end  # module BNPStep