module BNPStep

"""
Main module for BNP-Step.

This module serves as the entry point for running BNP-Step analysis.
"""

include("BNPAnalysis.jl")
using .BNPAnalysis
include("BNPInputs.jl")
using .BNPInputs



include("BNPSampler.jl")
using .BNPSampler

using Random
using LinearAlgebra
using DelimitedFiles  # Required for reading files
using StatsBase  # For statistical operations

# using GLMakie      # For visualization
using Distributions  # For sampling distributions



using HDF5
# using GLMakie
using Random, HDF5

"""
    simulate_and_save_ground_truth(filename::String; N::Int=1000, B::Int=20, noise_level::Float32=0.1f0)

Simulates synthetic step data with ground truth parameters, saves to `filename`, and returns the dataset and ground truth dictionary.
"""
function simulate_and_save_ground_truth(filename::String; N::Int=10000, B::Int=50, noise_level::Float32=0.25f0, write_data::Bool = true)
    t = collect(1:N)
    t_f32 = Float32.(t)

    # Ground truth parameters
    f_bg = 5.0f0
    h_m = Float32.((rand(Float32, B) .* 2f0) .+ 1f0)
    b_m = falses(B)
    b_m[1:25] .= true
    t_m = Float32.(sort(rand(1:N-100, B)))
    dt = 50.0f0
    eta = 1.0f0

    # Use helper to reconstruct signal
    signal = reconstruct_signal_from_sample( b_m, h_m, t_m, dt, f_bg, kernel,t_f32)

    noise = noise_level .* randn(Float32, N)
    data = signal .+ noise

    if write_data
        h5open(filename, "w") do file
            file["data"] = data
            file["times"] = t_f32
        end
    end

    dataset = Dict("data" => data, "times" => t_f32)
    ground_truth = Dict(
        "f_bg" => f_bg,
        "h_m" => h_m,
        "b_m" => b_m,
        "t_m" => Float32.(t_m),
        "dt"  => dt,
        "eta" => eta
    )

    return dataset, ground_truth
end





"""
    save_results(filename::String, results::Dict)

Save BNP-Step results to an HDF5 file.
"""
function save_results(filename::String, results::AbstractDict{String, <:Any})
    h5open(filename, "w") do file
        for (key, val) in results
            
            try
                file[key] = val  # Try writing directly
            catch e
                if isa(val, AbstractVector{<:AbstractVector})
                    lengths = map(length, val)
                    if all(x -> x == lengths[1], lengths)
                        # Uniform shape — convert to matrix
                        file[key] = Array(reduce(hcat, val)')
                    else
                        # Irregular shape — save as group of datasets
                        g = create_group(file, key)
                        for (i, subv) in enumerate(val)
                            g[string(i)] = subv
                        end
                    end
                else
                    rethrow(e)
                end
            end
        end
        @show keys(file)
    end
end


"""
    load_results(filename::String) -> Dict

Load BNP-Step results from an HDF5 file.
"""
function load_results(filename::String)::Dict{String, Vector}
    result = Dict{String, Any}()
    h5open(filename, "r") do file
        for key in keys(file)
            resval = read(file[key])
            result[key] = ndims(resval) == 1 ? vec(resval) : [resval[k, :] for k in axes(resval)[1]]
        end
    end
    return result
end



abstract type AbstractStep end
"""
    struct BNPStep

Defines the BNPStep structure for running BNP-Step analysis.

# Fields
- `chi::Float32`: Precision for Normal priors on h_m.
- `dt_ref::Float32`: Reference duration for Gamma prior on step durations.
- `h_ref::Float32`: Mean for Normal priors on h_m.
- `psi::Float32`: Precision for Normal prior on F_bg.
- `F_ref::Float32`: Mean for Normal prior on F_bg.
- `phi::Float32`: Shape parameter for Gamma prior on eta.
- `eta_ref::Float32`: Scale parameter for Gamma prior on eta.
- `gamma::Float32`: Hyperparameter for Bernoulli priors on b_m.
- `B_max::Int`: Maximum possible number of steps.
- `load_initialization::String`: Initialization strategy for b_m.
- `use_annealing::Bool`: Whether to use simulated annealing.
- `init_temperature::Int`: Initial temperature for simulated annealing.
- `scale_factor::Float32`: Controls how fast the temperature drops off.
- `rng::Random.AbstractRNG`: Random number generator.
"""
@kwdef struct BNP_Step<:AbstractStep
    chi::Float32
    dt_ref::Float32
    h_ref::Float32
    psi::Float32
    F_ref::Float32
    phi::Float32
    eta_ref::Float32
    gamma::Float32
    B_max::Int
    load_initialization::Symbol = :prior
    use_annealing::Bool
    init_temperature::Float32
    scale_factor::Float32
    rng::Random.AbstractRNG = Random.TaskLocalRNG()
    truth::Union{Dict{String,Any}, Nothing} = nothing
end


"""
    BNP_Step(; kwargs...)

Constructor for the BNPStep structure with default values.

# Keyword Arguments
- `chi::Float32`: Default is 0.028.
- `dt_ref::Float32`: Default is 100.0.
- `h_ref::Float32`: Default is 10.0.
- `psi::Float32`: Default is 0.0028.
- `F_ref::Float32`: Default is 0.0.
- `phi::Float32`: Default is 1.0.
- `eta_ref::Float32`: Default is 10.0.
- `gamma::Float32`: Default is 1.0.
- `B_max::Int`: Default is 50.
- `load_initialization::Symbol`: Default is "prior".
- `use_annealing::Bool`: Default is false.
- `init_temperature::Int`: Default is 2250.
- `scale_factor::Float32`: Default is 1.25.
- `seed::Union{Int, Nothing}`: Default is nothing.
"""
function BNP_Step_(; chi =0.028, 
                dt_ref  =100.0, 
                h_ref   =10.0,
                psi     =0.0028,
                F_ref   =0.0, 
                phi     =1.0, 
                eta_ref =10.0,
                gamma   =1.0, 
                B_max   =50, 
                load_initialization=:prior, 
                use_annealing=false,
                init_temperature=2250, 
                scale_factor=1.25, 
                rng=nothing, 
                truth=nothing)
    rng = isnothing(rng) ? Random.GLOBAL_RNG : MersenneTwister(rng)
    truth = isnothing(truth) ? simulate_and_save_ground_truth("";write_data=false)[2] : truth
    return BNP_Step(chi, dt_ref, h_ref, psi, F_ref, phi, eta_ref, gamma, B_max,
                   load_initialization, use_annealing, init_temperature, scale_factor, rng, truth)
end

function analyze(_step::AbstractStep, data::Dict, num_samples::Int=50000; freeze_loads::Bool=false, freeze_eta::Union{Float32,Bool}=true)
    # === Validate input data
    if !haskey(data, "data") || !haskey(data, "times")
        error("Dataset must contain 'data' and 'times' keys.")
    end

    data_points = data["data"]
    data_times = data["times"]
    num_data = length(data_points)
    t_n = data_times !== nothing ? data_times : collect(1:num_data)

    # === Fixed parameters
    nu_vec = ones(Float32, num_data)  # observation precision (Poisson + read noise)
    temp = 1.0f0               # no annealing for now

    # === Use ground truth initialization if requested
    if _step.load_initialization == :ground_truth && _step isa BNP_Step && _step.truth !== nothing
        b_m = _step.truth["b_m"]
        h_m = _step.truth["h_m"]
        t_m = _step.truth["t_m"]
        f_bg = _step.truth["f_bg"][end]
        eta = _step.truth["eta"][end]
        dt = _step.truth["dt"][end]
            # === Initialize traces (each as a Vector of previous samples)

    else
        if freeze_loads
            b_m = falses(_step.B_max)
        else
            b_m = trues(_step.B_max)
        end
        h_m = rand(Normal(_step.h_ref, sqrt(1 / _step.chi)), _step.B_max)
        t_m = rand(t_n, _step.B_max)
        f_bg = rand(Normal(_step.F_ref, sqrt(1 / _step.psi)))
        if freeze_eta
            eta = 8.079808
        else
            eta = rand(Gamma(_step.eta_ref / _step.phi, _step.phi))
        end
        dt = rand(Gamma(_step.dt_ref / _step.chi, _step.chi))
        # === Initialize traces (each as a Vector of previous samples)


        
    end

    @info "Initial posterior" calculate_logposterior(
        _step.B_max, num_data, data_points, t_n,
        [b_m], [h_m], [t_m], [dt], [f_bg], [eta], nu_vec,
        _step.chi, _step.h_ref, _step.gamma, _step.phi, _step.eta_ref,
        _step.psi, _step.dt_ref, _step.F_ref, kernel)
    # === Results dictionary (strong typing)
    results = Dict(
        "b_m" => Vector{Vector{Bool}}(),
        "h_m" => Vector{Vector{Float32}}(),
        "t_m" => Vector{Vector{Float32}}(),
        "f_bg" => Vector{Float32}(),
        "eta" => Vector{Float32}(),
        "dt" => Vector{Float32}(),
        "posterior" => Vector{Float32}()
    )

    b_m_trace = [b_m]
    h_m_trace = [h_m]
    t_m_trace = [t_m]
    f_bg_trace = [f_bg]
    eta_trace = [eta]
    dt_trace = [dt]

    # === Gibbs sampling loop
    for _ in 1:num_samples
        fh = sample_fh(_step.B_max, num_data, data_points, t_n, b_m_trace, t_m_trace, dt_trace, eta_trace, nu_vec,
                       _step.psi, _step.chi, _step.F_ref, _step.h_ref, _step.rng, temp, kernel)
        f_bg = fh[1]
        h_m = fh[2:end]
        push!(results["h_m"], h_m)
        push!(h_m_trace, h_m)
        push!(results["f_bg"], f_bg)
        push!(f_bg_trace, f_bg)
        if freeze_loads
            b_m = falses(_step.B_max)
        else
            b_m = sample_b(_step.B_max, num_data, data_points, t_n, b_m_trace, h_m_trace, t_m_trace, dt_trace,
                        f_bg_trace, eta_trace, nu_vec, _step.gamma, _step.rng, temp, kernel)
        end
        push!(results["b_m"], b_m)
        push!(b_m_trace, b_m)

        if !freeze_loads 
            t_m = sample_t(_step.B_max, num_data, data_points, t_n, b_m_trace, h_m_trace, t_m_trace, dt_trace,
                       f_bg_trace, eta_trace, nu_vec, _step.rng, temp, kernel)
        end
        push!(results["t_m"], t_m)
        push!(t_m_trace, t_m)
        if freeze_eta
            eta = 8.079808
        else
            eta = sample_eta_metropolis(eta_trace, nu_vec, _step.B_max, num_data, data_points, t_n,
                                        b_m_trace, h_m_trace, t_m_trace, dt_trace, f_bg_trace,
                                        _step.phi, _step.eta_ref, _step.rng, temp, kernel)
        end
        push!(results["eta"], eta)
        push!(eta_trace, eta)

        if !freeze_loads 
            dt = sample_dt_metropolis(eta_trace, nu_vec, _step.B_max, num_data, data_points, t_n,
                                   b_m_trace, h_m_trace, t_m_trace, dt_trace, f_bg_trace,
                                   _step.chi, _step.dt_ref, _step.rng, temp, kernel)
        end
        push!(results["dt"], dt)
        push!(dt_trace, dt)

        post, _ = calculate_logposterior(_step.B_max, num_data, data_points, t_n,
                                          b_m_trace, h_m_trace, t_m_trace, dt_trace, f_bg_trace, eta_trace, nu_vec,
                                          _step.chi, _step.h_ref, _step.gamma, _step.phi, _step.eta_ref,
                                          _step.psi, _step.dt_ref, _step.F_ref, kernel)

                                          # === Append to results and traces
        push!(results["posterior"], post)


    end

    return results
end

# function visualize_results(results::Dict, data::Dict; plot_type::String="step", font_size::Int=16,
#     datacolor::Symbol=:gray, learncolor::Symbol=:orange, truth=nothing)
    

#     # Extract data
#     data_points = data["data"]
#     data_times = data["times"]
#     t_n = data_times !== nothing ? data_times : collect(1:length(data_points))

#     # Extract results
#     b_m = results["b_m"]
#     h_m = results["h_m"]
#     t_m = results["t_m"]
#     f_bg = results["f_bg"]
#     eta = results["eta"]
#     dt = results["dt"]
#     posterior = results["posterior"]

#     if plot_type == "step"
#         println("Generating step plot with heatmap...")

#         # === MAP sample
#         map_idx = argmax(posterior)
#         step_map = reconstruct_signal_from_sample(b_m[map_idx], h_m[map_idx], t_m[map_idx], dt[map_idx], f_bg[map_idx], kernel, t_n)

#         # === Last sample (if not same as MAP)
#         last_idx = length(posterior)
#         step_last = last_idx != map_idx ?
#         reconstruct_signal_from_sample(b_m[last_idx], h_m[last_idx], t_m[last_idx], dt[last_idx], f_bg[last_idx], kernel, t_n) :
#         nothing

#         # === Top-K heatmap (from partial posterior)
#         topK = 100
#         top_inds = partialsortperm(posterior, rev=true, 1:topK)
#         traces = [reconstruct_signal_from_sample(b_m[i], h_m[i], t_m[i], dt[i], f_bg[i], kernel, t_n) for i in top_inds]
#         S = hcat(traces...)'  # [topK × T]
#         T = repeat(t_n', topK, 1)  # [topK × T]
#         points = (vec(T), vec(S))

#         h2d = fit(Histogram2D, points..., nbins=(length(t_n) ÷ 4, 100))
#         H = h2d.weights ./ maximum(h2d.weights)

#         # === Plot heatmap
#         heatmap(h2d.edges[1], h2d.edges[2], H';
#         color=:gray, alpha=0.4, colorbar=false, label="", legend=:topright)

#         # === Plot curves
#         plot!(t_n, data_points, label="Data", color=datacolor, lw=1.5)
#         plot!(t_n, step_map, label="Steps (MAP)", color=learncolor, lw=2.0)
#         if step_last !== nothing
#         plot!(t_n, step_last, label="Steps (Last Sample)", color=:blue, lw=1.5, linestyle=:dash)
#         end
#         if truth !== nothing
#         gt = reconstruct_signal(t_n, truth["b_m"], truth["h_m"], truth["t_m"], truth["dt"], truth["f_bg"], kernel)
#         plot!(t_n, gt, label="Ground Truth", color=:green, lw=1.5, linestyle=:dot)
#         end

#         xlabel!("Time", fontsize=font_size)
#         ylabel!("Signal", fontsize=font_size)
#         title!("BNP-Step Trajectory Posterior", fontsize=font_size + 2)

#     elseif plot_type == "hist_step_height"
#         println("Generating histogram of step heights...")

#         # Flatten step heights for histogram
#         step_heights = vcat(h_m...)
#         histogram(step_heights, bins=20, label="Step Heights", color=learncolor, alpha=0.7, legend=:topright)
#         xlabel!("Step Height")
#         ylabel!("Frequency")
#         title!("Histogram of Step Heights")
#     elseif plot_type == "hist_dwell_time"
#         println("Generating histogram of dwell times...")

#         # Compute dwell times
#         dwell_times = []
#         for i in eachindex(b_m)
#             active_indices = findall(b_m[i])
#             append!(dwell_times, diff(vcat(0.0, t_m[i][active_indices], maximum(t_n))))
#         end

#         # Plot histogram
#         histogram(dwell_times, bins=20, label="Dwell Times", color=learncolor, alpha=0.7, legend=:topright)
#         xlabel!("Dwell Time")
#         ylabel!("Frequency")
#         title!("Histogram of Dwell Times")
#     elseif plot_type == "hist_emission"
#         println("Generating histogram of emission levels...")

#         # Flatten emission levels for histogram
#         emission_levels = vcat(f_bg .+ h_m...)
#         histogram(emission_levels, bins=20, label="Emission Levels", color=learncolor, alpha=0.7, legend=:topright)
#         xlabel!("Emission Level")
#         ylabel!("Frequency")
#         title!("Histogram of Emission Levels")
#     elseif plot_type == "hist_height_separated"
#         println("Generating separated histogram of step heights...")

#         # Separate step heights by active Bernoulli variables
#         for i in eachindex(b_m)
#             active_heights = h_m[i][b_m[i]]
#             histogram(active_heights, bins=10, label="Step Heights (Sample $i)", alpha=0.5, lw=1.5, legend=:topright)
#         end
#         xlabel!("Step Height")
#         ylabel!("Frequency")
#         title!("Separated Histogram of Step Heights")
#     elseif plot_type == "hist_f"
#         println("Generating histogram of background fluorescence (F_bg)...")

#         # Plot histogram of F_bg
#         histogram(f_bg, bins=20, label="F_bg", color=learncolor, alpha=0.7, legend=:topright)
#         xlabel!("Background Fluorescence (F_bg)")
#         ylabel!("Frequency")
#         title!("Histogram of Background Fluorescence")
#     elseif plot_type == "hist_eta"
#         println("Generating histogram of noise parameters (eta)...")

#         # Plot histogram of eta
#         histogram(eta, bins=20, label="Eta", color=learncolor, alpha=0.7, legend=:topright)
#         xlabel!("Noise Parameter (Eta)")
#         ylabel!("Frequency")
#         title!("Histogram of Noise Parameters")
#     elseif plot_type == "survivorship"
#         println("Generating survivorship plot...")

#         # Compute dwell times
#         dwell_times = []
#         for i in 1:length(b_m)
#             active_indices = findall(b_m[i])
#             append!(dwell_times, diff(vcat(0.0, t_m[i][active_indices], maximum(t_n))))
#         end

#         # Sort dwell times and compute survivorship
#         sorted_dwell_times = sort(dwell_times)
#         survivorship = 1 .- cumsum(fill(1.0 / length(sorted_dwell_times), length(sorted_dwell_times)))

#         # Plot survivorship curve
#         plot(sorted_dwell_times, survivorship, label="Survivorship", color=learncolor, lw=2.0, legend=:topright)
#         xlabel!("Dwell Time")
#         ylabel!("Survivorship")
#         title!("Survivorship Plot")
#     elseif plot_type == "hist_dwell_separated"
#         println("Generating separated histogram of dwell times...")

#         # Separate dwell times by active Bernoulli variables
#         for i in 1:length(b_m)
#             active_indices = findall(b_m[i])
#             dwell_times = diff(vcat(0.0, t_m[i][active_indices], maximum(t_n)))
#             histogram(dwell_times, bins=10, label="Dwell Times (Sample $i)", alpha=0.5, lw=1.5, legend=:topright)
#         end
#         xlabel!("Dwell Time")
#         ylabel!("Frequency")
#         title!("Separated Histogram of Dwell Times")
#     elseif plot_type == "hist_emission_separated"
#         println("Generating separated histogram of emission levels...")

#         # Separate emission levels by active Bernoulli variables
#         for i in 1:length(b_m)
#             active_emissions = f_bg + h_m[i][b_m[i]]
#             histogram(active_emissions, bins=10, label="Emission Levels (Sample $i)", alpha=0.5, lw=1.5, legend=:topright)
#         end
#         xlabel!("Emission Level")
#         ylabel!("Frequency")
#         title!("Separated Histogram of Emission Levels")
#     else
#         println("Unsupported plot type: $plot_type")
#     end
# end

"""
    load_data(filename::String; kwargs...) -> Dict

Loads a dataset for BNP-Step analysis.
"""
function load_data(filename::String; kwargs...)
    # Implement logic to load data based on file type and options
    return Dict("data" => rand(100), "times" => collect(1:100))  # Example placeholder
end

"""
    BNP_Step_from_ground_truth(truth::Dict{String,Any})

Construct a BNP_Step instance using values from a ground truth dictionary.
"""
function BNP_Step_from_ground_truth(truth::Dict{String,Any})
    return BNP_Step_(
        chi = 0.001f0,
        dt_ref = truth["dt"],
        h_ref = mean(vcat(truth["h_m"]...)),
        psi = 1.0f0,
        F_ref = truth["f_bg"],
        phi = 1.0f0,
        eta_ref = truth["eta"],
        gamma = 1.0f0,
        B_max = length(truth["b_m"]),
        load_initialization = :ground_truth,
        use_annealing = false,
        init_temperature = 1,
        scale_factor = 1.0f0,
        # rng = MersenneTwister(6626),  # Fixed seed for reproducibility
        truth = Dict(
            "b_m" => truth["b_m"],
            "h_m" => truth["h_m"],
            "t_m" => truth["t_m"],
            "f_bg" => truth["f_bg"],
            "eta" => truth["eta"],
            "dt" => truth["dt"]
        )
    )
end


export save_results, load_results, load_data,
 AbstractStep, BNP_Step, analyze, kernel, simulate_and_save_ground_truth,
 BNP_Step_from_ground_truth, reconstruct_signal_from_trace
export load_data_garcia, load_data_txt, load_data_csv,
 load_data_HMM, load_data_expt, load_data_kv, reconstruct_signal_from_trace
export sample_b, sample_fh, sample_t, sample_eta_metropolis, sample_dt_metropolis, 
 calculate_loglikelihood, calculate_logposterior, emit_results_snapshot, BNP_Step_


end  # module BNPStep