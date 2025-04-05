module BNPAnalysis

using CSV
using DataFrames
using LinearAlgebra
using Statistics

"""
    calculate_step_statistics(data::Vector{Float64}, times::Vector{Float64})

Calculates step statistics such as mean, variance, and step duration.

# Arguments
- `data::Vector{Float64}`: Observed data points.
- `times::Vector{Float64}`: Corresponding time points.

# Returns
A dictionary containing step statistics.
"""
function calculate_step_statistics(data::Vector{Float64}, times::Vector{Float64})
    if length(data) != length(times)
        throw(ArgumentError("Data and times must have the same length."))
    end

    step_durations = diff(times)
    mean_step = mean(data)
    variance_step = var(data)

    return Dict(
        "mean_step" => mean_step,
        "variance_step" => variance_step,
        "step_durations" => step_durations
    )
end

"""
    detect_transitions(data::Vector{Float64}, threshold::Float64)

Detects transitions in the data based on a threshold.

# Arguments
- `data::Vector{Float64}`: Observed data points.
- `threshold::Float64`: Threshold for detecting transitions.

# Returns
A vector of indices where transitions occur.
"""
function detect_transitions(data::Vector{Float64}, threshold::Float64)
    transitions = findall(x -> abs(x) > threshold, diff(data))
    return transitions
end

"""
    analyze_step_data(data::Vector{Float64}, times::Vector{Float64}, threshold::Float64)

Performs a complete analysis of step data, including transition detection and step statistics.

# Arguments
- `data::Vector{Float64}`: Observed data points.
- `times::Vector{Float64}`: Corresponding time points.
- `threshold::Float64`: Threshold for detecting transitions.

# Returns
A dictionary containing transition indices and step statistics.
"""
function analyze_step_data(data::Vector{Float64}, times::Vector{Float64}, threshold::Float64)
    transitions = detect_transitions(data, threshold)
    stats = calculate_step_statistics(data, times)

    return Dict(
        "transitions" => transitions,
        "statistics" => stats
    )
end

"""
    calculate_transition_rates(transitions::Vector{Int}, times::Vector{Float64})

Calculates the transition rates based on detected transitions and corresponding time points.

# Arguments
- `transitions::Vector{Int}`: Indices of detected transitions.
- `times::Vector{Float64}`: Corresponding time points.

# Returns
The transition rates as a vector of Float64.
"""
function calculate_transition_rates(transitions::Vector{Int}, times::Vector{Float64})
    if isempty(transitions)
        return Float64[]
    end

    rates = Float64[]
    for i in 2:length(transitions)
        push!(rates, 1.0 / (times[transitions[i]] - times[transitions[i - 1]]))
    end

    return rates
end

"""
    calculate_step_durations(transitions::Vector{Int}, times::Vector{Float64})

Calculates the durations of steps based on detected transitions and corresponding time points.

# Arguments
- `transitions::Vector{Int}`: Indices of detected transitions.
- `times::Vector{Float64}`: Corresponding time points.

# Returns
A vector of step durations as Float64.
"""
function calculate_step_durations(transitions::Vector{Int}, times::Vector{Float64})
    if isempty(transitions)
        return Float64[]
    end

    durations = Float64[]
    for i in 2:length(transitions)
        push!(durations, times[transitions[i]] - times[transitions[i - 1]])
    end

    return durations
end

"""
    calculate_mean_transition_rate(transitions::Vector{Int}, times::Vector{Float64})

Calculates the mean transition rate based on detected transitions and corresponding time points.

# Arguments
- `transitions::Vector{Int}`: Indices of detected transitions.
- `times::Vector{Float64}`: Corresponding time points.

# Returns
The mean transition rate as a Float64.
"""
function calculate_mean_transition_rate(transitions::Vector{Int}, times::Vector{Float64})
    if isempty(transitions) || length(transitions) < 2
        return 0.0
    end

    durations = calculate_step_durations(transitions, times)
    mean_rate = 1.0 / mean(durations)
    return mean_rate
end

"""
    calculate_variance_of_steps(data::Vector{Float64}, transitions::Vector{Int})

Calculates the variance of steps based on detected transitions.

# Arguments
- `data::Vector{Float64}`: Observed data points.
- `transitions::Vector{Int}`: Indices of detected transitions.

# Returns
A vector of variances for each step.
"""
function calculate_variance_of_steps(data::Vector{Float64}, transitions::Vector{Int})
    if isempty(transitions) || length(transitions) < 2
        return Float64[]
    end

    variances = Float64[]
    for i in 2:length(transitions)
        step_data = data[transitions[i - 1]:transitions[i] - 1]
        push!(variances, var(step_data))
    end

    return variances
end

"""
    calculate_step_means(data::Vector{Float64}, transitions::Vector{Int})

Calculates the mean of each step based on detected transitions.

# Arguments
- `data::Vector{Float64}`: Observed data points.
- `transitions::Vector{Int}`: Indices of detected transitions.

# Returns
A vector of means for each step.
"""
function calculate_step_means(data::Vector{Float64}, transitions::Vector{Int})
    if isempty(transitions) || length(transitions) < 2
        return Float64[]
    end

    means = Float64[]
    for i in 2:length(transitions)
        step_data = data[transitions[i - 1]:transitions[i] - 1]
        push!(means, mean(step_data))
    end

    return means
end

"""
    load_ihmm_mode_means(filename::String, path::Union{String, Nothing}=nothing)

Loads the mean emission levels from all samples with the mode number of states
generated from the iHMM method.

# Arguments
- `filename::String`: Name of the file to be loaded.
- `path::Union{String, Nothing}`: Path where the file is located.

# Returns
A vector containing the mode means from the iHMM.
"""
function load_ihmm_mode_means(filename::String, path::Union{String, Nothing}=nothing)
    if !(typeof(filename) <: String)
        throw(TypeError("filename should be of type String"))
    end

    full_name = filename * ".csv"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    heights = Float64[]
    for row in CSV.File(full_path)
        append!(heights, parse.(Float64, row))
    end

    return heights
end

"""
    load_ihmm_mode_mean_trajectory(filename::String, path::Union{String, Nothing}=nothing)

Loads mode mean trajectory from iHMM method.

# Arguments
- `filename::String`: Name of the file to be loaded.
- `path::Union{String, Nothing}`: Path where the file is located.

# Returns
A vector containing the mode mean trajectory.
"""
function load_ihmm_mode_mean_trajectory(filename::String, path::Union{String, Nothing}=nothing)
    if !(typeof(filename) <: String)
        throw(TypeError("filename should be of type String"))
    end

    full_name = filename * ".csv"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    sampled_heights = Float64[]
    for row in CSV.File(full_path)
        append!(sampled_heights, parse.(Float64, row))
    end

    return sampled_heights
end

"""
    load_ihmm_samples(filename::String, skip_indices::String, path::Union{String, Nothing}=nothing)

Loads generated samples from the iHMM method. Only those samples with the mode number of states are selected.

# Arguments
- `filename::String`: Name of the file to be loaded.
- `skip_indices::String`: File with the list of samples to skip.
- `path::Union{String, Nothing}`: Path where the file is located.

# Returns
A matrix containing the mode mean trajectory samples.
"""
function load_ihmm_samples(filename::String, skip_indices::String, path::Union{String, Nothing}=nothing)
    if !(typeof(filename) <: String)
        throw(TypeError("filename should be of type String"))
    end
    if !(typeof(skip_indices) <: String)
        throw(TypeError("skip_indices should be of type String"))
    end

    full_name = filename * ".csv"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    ind_name = skip_indices * ".csv"
    ind_path = isnothing(path) ? ind_name : joinpath(path, ind_name)

    # Read skip indices
    skip_ind = Int[]
    for row in CSV.File(ind_path)
        append!(skip_ind, parse.(Int, row))
    end

    # Read samples
    samples = Float64[]
    cur_ind = 1
    ind_count = 1
    for row in CSV.File(full_path)
        if cur_ind <= length(skip_ind) && ind_count == skip_ind[cur_ind]
            cur_ind += 1
            ind_count += 1
            continue
        end
        append!(samples, parse.(Float64, row))
        ind_count += 1
    end

    return reshape(samples, :, ind_count - cur_ind)
end

"""
    parallel_bubble_sort(times::Vector{Float64}, data::Vector{Float64})

Sorts data sets with time points in chronological order.

# Arguments
- `times::Vector{Float64}`: Array of time points.
- `data::Vector{Float64}`: Array of observations.

# Returns
A tuple of sorted time points and sorted observations.
"""
function parallel_bubble_sort(times::Vector{Float64}, data::Vector{Float64})
    num_times = length(times)
    for i in 1:num_times
        done_sorting = true
        for j in 1:(num_times - i)
            if times[j] > times[j + 1]
                times[j], times[j + 1] = times[j + 1], times[j]
                data[j], data[j + 1] = data[j + 1], data[j]
                done_sorting = false
            end
        end
        if done_sorting
            break
        end
    end

    return times, data
end

"""
    get_credible_intervals(states::Vector{Float64})

Calculates the credible intervals associated with an array.

# Arguments
- `states::Vector{Float64}`: Array of values for which CI's will be calculated.

# Returns
A tuple containing mean, 95% lower bound, 50% lower bound, median, 50% upper bound, and 95% upper bound.
"""
function get_credible_intervals(states::Vector{Float64})
    mean = mean(states)
    under95 = quantile(states, 0.025)
    under50 = quantile(states, 0.25)
    median = quantile(states, 0.5)
    upper50 = quantile(states, 0.75)
    upper95 = quantile(states, 0.975)
    return mean, under95, under50, median, upper50, upper95
end

"""
    remove_burn_in(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                   dt_vec::Matrix{Float64}, f_vec::Vector{Float64}, eta_vec::Vector{Float64}, 
                   post_vec::Vector{Float64}, n::Int)

Removes the first n samples (burn-in).

# Arguments
- `b_vec::Matrix{Float64}`: Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
- `h_vec::Matrix{Float64}`: Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
- `t_vec::Matrix{Float64}`: Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
- `dt_vec::Matrix{Float64}`: Array of samples of the step durations.
- `f_vec::Vector{Float64}`: Array of samples of the background F_bg.
- `eta_vec::Vector{Float64}`: Array of samples of the noise variance eta.
- `post_vec::Vector{Float64}`: Array of the calculated log posterior for each sample.
- `n::Int`: Number of burn-in samples to discard.

# Returns
A tuple containing cleaned arrays with burn-in removed.
"""
function remove_burn_in(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                        dt_vec::Matrix{Float64}, f_vec::Vector{Float64}, eta_vec::Vector{Float64}, 
                        post_vec::Vector{Float64}, n::Int)
    b_vec_clean = b_vec[n+1:end, :]
    h_vec_clean = h_vec[n+1:end, :]
    t_vec_clean = t_vec[n+1:end, :]
    dt_vec_clean = dt_vec[n+1:end, :]
    f_vec_clean = f_vec[n+1:end]
    eta_vec_clean = eta_vec[n+1:end]
    post_vec_clean = post_vec[n+1:end]

    return b_vec_clean, h_vec_clean, t_vec_clean, dt_vec_clean, f_vec_clean, eta_vec_clean, post_vec_clean
end

"""
    find_map(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
             dt_vec::Matrix{Float64}, f_vec::Vector{Float64}, eta_vec::Vector{Float64}, 
             posteriors::Vector{Float64})

Locates the maximum a posteriori (MAP) estimate sample from the results returned by BNP-Step.

# Arguments
- `b_vec::Matrix{Float64}`: Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
- `h_vec::Matrix{Float64}`: Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
- `t_vec::Matrix{Float64}`: Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
- `dt_vec::Matrix{Float64}`: Array of samples of the step durations.
- `f_vec::Vector{Float64}`: Array of samples of the background F_bg.
- `eta_vec::Vector{Float64}`: Array of samples of the noise variance eta.
- `posteriors::Vector{Float64}`: Array of the calculated log posterior for each sample.

# Returns
A tuple containing the MAP estimate values for b_m, h_m, t_m, dt_m, F_bg, and eta.
"""
function find_map(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                  dt_vec::Matrix{Float64}, f_vec::Vector{Float64}, eta_vec::Vector{Float64}, 
                  posteriors::Vector{Float64})
    map_index = argmax(posteriors)
    f_clean = f_vec[map_index]
    b_clean = b_vec[map_index, :]
    h_clean = h_vec[map_index, :]
    t_clean = t_vec[map_index, :]
    dt_clean = dt_vec[map_index, :]
    eta_clean = isempty(eta_vec) ? 0.0 : eta_vec[map_index]

    return b_clean, h_clean, t_clean, dt_clean, f_clean, eta_clean
end

"""
    find_top_n_samples(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                       f_vec::Vector{Float64}, eta_vec::Vector{Float64}, posteriors::Vector{Float64}, 
                       weak_limit::Int, num_samples::Int=10)

Picks out the top n samples (specified by the user) from those generated by BNP-Step, regardless of the number of steps.

# Arguments
- `b_vec::Matrix{Float64}`: Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
- `h_vec::Matrix{Float64}`: Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
- `t_vec::Matrix{Float64}`: Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
- `f_vec::Vector{Float64}`: Array of samples of the background F_bg.
- `eta_vec::Vector{Float64}`: Array of samples of the noise variance eta.
- `posteriors::Vector{Float64}`: Array of the calculated log posterior for each sample.
- `weak_limit::Int`: Maximum number of possible steps.
- `num_samples::Int`: Number of top samples to return.

# Returns
A tuple containing the top samples for b_m, h_m, t_m, F_bg, and eta.
"""
function find_top_n_samples(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                            f_vec::Vector{Float64}, eta_vec::Vector{Float64}, posteriors::Vector{Float64}, 
                            weak_limit::Int, num_samples::Int=10)
    f_top = zeros(Float64, num_samples)
    eta_top = zeros(Float64, num_samples)
    b_m_top = zeros(Float64, num_samples, weak_limit)
    h_m_top = zeros(Float64, num_samples, weak_limit)
    t_m_top = zeros(Float64, num_samples, weak_limit)

    # Sort indices in descending order of posteriors
    sorting_indices = sortperm(posteriors, rev=true)

    for i in 1:num_samples
        f_top[i] = f_vec[sorting_indices[i]]
        eta_top[i] = eta_vec[sorting_indices[i]]
        b_m_top[i, :] = b_vec[sorting_indices[i], :]
        h_m_top[i, :] = h_vec[sorting_indices[i], :]
        t_m_top[i, :] = t_vec[sorting_indices[i], :]
    end

    return b_m_top, h_m_top, t_m_top, f_top, eta_top
end

"""
    find_top_samples_by_jumps(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                              f_vec::Vector{Float64}, eta_vec::Vector{Float64}, posteriors::Vector{Float64})

Picks out all samples with the MAP number of steps.

# Arguments
- `b_vec::Matrix{Float64}`: Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
- `h_vec::Matrix{Float64}`: Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
- `t_vec::Matrix{Float64}`: Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
- `f_vec::Vector{Float64}`: Array of samples of the background F_bg.
- `eta_vec::Vector{Float64}`: Array of samples of the noise variance eta.
- `posteriors::Vector{Float64}`: Array of the calculated log posterior for each sample.

# Returns:
A tuple containing the samples for b_m, h_m, t_m, F_bg, and eta with the MAP number of steps.
"""
function find_top_samples_by_jumps(b_vec::Matrix{Float64}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                                   f_vec::Vector{Float64}, eta_vec::Vector{Float64}, posteriors::Vector{Float64})
    # Determine MAP number of jumps
    map_index = argmax(posteriors)
    map_jump_number = sum(b_vec[map_index, :])

    # Filter samples with the MAP number of steps
    num_jumps = sum(b_vec, dims=2)
    good_indices = findall(x -> x == map_jump_number, num_jumps)

    good_b_m = b_vec[good_indices, :]
    good_h_m = h_vec[good_indices, :]
    good_t_m = t_vec[good_indices, :]
    good_f_s = f_vec[good_indices]
    good_eta = eta_vec[good_indices]

    return good_b_m, good_h_m, good_t_m, good_f_s, good_eta
end

"""
    generate_step_plot_data(b_vec::Vector{Int}, h_vec::Vector{Float64}, t_vec::Vector{Float64}, 
                            dt_vec::Vector{Float64}, f_vec::Float64, weak_limit::Int, 
                            t_n::Vector{Float64}, kernel::Function)

Generates a plottable trajectory from a MAP estimate sample from BNP-Step.

# Arguments
- `b_vec::Vector{Int}`: Array of MAP estimate b_m from a BNP-Step sample.
- `h_vec::Vector{Float64}`: Array of MAP estimate h_m from a BNP-Step sample.
- `t_vec::Vector{Float64}`: Array of MAP estimate t_m from a BNP-Step sample.
- `dt_vec::Vector{Float64}`: Array of MAP estimate step durations.
- `f_vec::Float64`: MAP estimate F_bg from a BNP-Step sample.
- `weak_limit::Int`: Maximum possible number of steps in the data set.
- `t_n::Vector{Float64}`: Array of time points for the trajectory.
- `kernel::Function`: Kernel function for step calculation.

# Returns
A tuple containing sorted time points and pseudo-observations for the trajectory.
"""
function generate_step_plot_data(b_vec::Vector{Int}, h_vec::Vector{Float64}, t_vec::Vector{Float64}, 
                                 dt_vec::Vector{Float64}, f_vec::Float64, weak_limit::Int, 
                                 t_n::Vector{Float64}, kernel::Function)
    # Count total number of transitions
    num_data = length(t_n)
    jmp_count = sum(b_vec)

    # Initialize clean arrays to store only 'on' loads
    sampled_loads = ones(Int, jmp_count)
    sampled_times = zeros(Float64, jmp_count)
    sampled_heights = zeros(Float64, jmp_count)

    # Strip out all the 'off' loads
    ind = 1
    for i in 1:weak_limit
        if b_vec[i] == 1
            sampled_heights[ind] = h_vec[i]
            sampled_times[ind] = t_vec[i]
            ind += 1
        end
    end

    # Pre-calculate matrices required for vectorized sum calculations
    times_matrix = repeat(sampled_times', num_data, 1)
    obs_time_matrix = repeat(t_n, 1, jmp_count)
    height_matrix = repeat(sampled_heights', num_data, 1)
    load_matrix = repeat(sampled_loads', num_data, 1)

    # Calculate product of b_m and h_m term-wise
    bh_matrix = load_matrix .* height_matrix

    # Reconstruct "data" based on our sampled values
    if isempty(dt_vec)
        bht_matrix = bh_matrix .* kernel(obs_time_matrix .- times_matrix)
    else
        bht_matrix = bh_matrix .* kernel(obs_time_matrix .- times_matrix, dt_vec)
    end

    # Calculate sum term - these are the pseudo-observations
    sampled_data = f_vec .+ sum(bht_matrix, dims=2)

    # Make arrays for graphing step plots
    sorted_times, sorted_data = parallel_bubble_sort(sampled_times, vec(sampled_data))
    sorted_times = vcat(t_n[1], sorted_times, t_n[end])

    return sorted_times, sorted_data
end

"""
    generate_gt_step_plot_data(ground_b_m::Vector{Int}, ground_h_m::Vector{Float64}, 
                               ground_t_m::Vector{Float64}, ground_f::Float64, 
                               data_times::Vector{Float64}, weak_limit::Int, kernel::Function)

Generates a ground truth trajectory for data sets where the ground truth is known.

# Arguments
- `ground_b_m::Vector{Int}`: Ground truth loads.
- `ground_h_m::Vector{Float64}`: Ground truth step heights.
- `ground_t_m::Vector{Float64}`: Ground truth step times.
- `ground_f::Float64`: Ground truth value for F_bg.
- `data_times::Vector{Float64}`: Array of time points for the trajectory.
- `weak_limit::Int`: Maximum possible number of steps in the data set.
- `kernel::Function`: Kernel function for step calculation.

# Returns
A tuple containing sorted time points and pseudo-observations for the ground truth trajectory.
"""
function generate_gt_step_plot_data(ground_b_m::Vector{Int}, ground_h_m::Vector{Float64}, 
                                    ground_t_m::Vector{Float64}, ground_f::Float64, 
                                    data_times::Vector{Float64}, weak_limit::Int, kernel::Function)
    # Count total number of jump points
    jmp_count_gnd = sum(ground_b_m)

    # Strip out all non-jump points
    ground_jumps = ones(Int, jmp_count_gnd)
    ground_times = zeros(Float64, jmp_count_gnd)
    ground_heights = zeros(Float64, jmp_count_gnd)
    ind = 1
    for i in 1:weak_limit
        if ground_b_m[i] == 1
            ground_heights[ind] = ground_h_m[i]
            ground_jumps[ind] = ground_b_m[i]
            ground_times[ind] = ground_t_m[i]
            ind += 1
        end
    end

    # Pre-calculate matrices required for vectorized sum calculations
    times_matrix = repeat(ground_times', jmp_count_gnd, 1)
    obs_time_matrix = repeat(ground_times, 1, jmp_count_gnd)
    height_matrix = repeat(ground_heights', jmp_count_gnd, 1)
    load_matrix = repeat(ground_jumps', jmp_count_gnd, 1)

    # Calculate product of b_m and h_m term-wise
    bh_matrix = load_matrix .* height_matrix

    # Reconstruct "data" based on our sampled values
    bht_matrix = bh_matrix .* kernel(obs_time_matrix .- times_matrix)

    # Calculate sum term - this is your sampled data
    ground_data = ground_f .+ sum(bht_matrix, dims=2)

    # Make arrays for graphing step plots
    sorted_times, sorted_data = parallel_bubble_sort(ground_times, vec(ground_data))
    sorted_times = vcat(data_times[1], sorted_times, data_times[end])
    sorted_data = vcat(sorted_data, ground_f)

    return sorted_times, sorted_data
end

"""
    generate_kv_step_plot_data(jump_times::Vector{Float64}, heights::Vector{Float64}, 
                               background::Float64, data_times::Vector{Float64})

Generates step plot data from BIC-based method results.

# Arguments
- `jump_times::Vector{Float64}`: Array of jump times returned by the BIC method.
- `heights::Vector{Float64}`: Array of inter-step means returned by the BIC method. Does not include the final mean.
- `background::Float64`: Value of the final inter-step mean returned by the BIC method.
- `data_times::Vector{Float64}`: Array of time points.

# Returns
A tuple containing the time points and pseudo-observations for the learned trajectory.
"""
function generate_kv_step_plot_data(jump_times::Vector{Float64}, heights::Vector{Float64}, 
                                    background::Float64, data_times::Vector{Float64})
    # Add the first observation time point to the start of the array and duplicate the end point
    plot_times = vcat(data_times[1], jump_times, data_times[end])

    # Append the background value
    plot_heights = vcat(heights, background)

    return plot_times, plot_heights
end

"""
    generate_histogram_data(b_vec::Matrix{Int}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                            num_samples::Int, weak_limit::Int, times::Vector{Float64})

Processes raw BNP-Step results into a format that can be histogrammed.

# Arguments
- `b_vec::Matrix{Int}`: Array of b_m from BNP-Step samples.
- `h_vec::Matrix{Float64}`: Array of h_m from BNP-Step samples.
- `t_vec::Matrix{Float64}`: Array of t_m from BNP-Step samples.
- `num_samples::Int`: Number of samples kept for histogramming.
- `weak_limit::Int`: Maximum possible number of steps in the data set.
- `times::Vector{Float64}`: Array of time points.

# Returns
A tuple containing arrays of absolute step heights and holding times between steps.
"""
function generate_histogram_data(b_vec::Matrix{Int}, h_vec::Matrix{Float64}, t_vec::Matrix{Float64}, 
                                 num_samples::Int, weak_limit::Int, times::Vector{Float64})
    histogram_heights = Float64[]
    histogram_lengths = Float64[]

    for i in 1:num_samples
        temp_times = Float64[]
        for j in 1:weak_limit
            if b_vec[i, j] == 1
                push!(histogram_heights, h_vec[i, j])
                push!(temp_times, t_vec[i, j])
            end
        end
        temp_times = sort(temp_times)
        temp_times = vcat(times[1], temp_times, times[end])
        for j in 2:length(temp_times)
            push!(histogram_lengths, temp_times[j] - temp_times[j - 1])
        end
    end

    return histogram_heights, histogram_lengths
end

"""
    generate_histogram_data_ihmm(samples::Matrix{Float64}, times::Vector{Float64})

Converts iHMM results to step height form for histogramming. Also returns the holding times.

# Arguments
- `samples::Matrix{Float64}`: Array of iHMM samples with the mode number of states.
- `times::Vector{Float64}`: Time points for each observation.

# Returns
A tuple containing arrays of absolute step heights and holding times between the steps.
"""
function generate_histogram_data_ihmm(samples::Matrix{Float64}, times::Vector{Float64})
    histogram_heights = Float64[]
    histogram_times = Float64[]

    num_samples = size(samples, 1)
    traj_len = size(samples, 2)

    for i in 1:num_samples
        time_prev = times[1]
        for j in 2:traj_len
            if samples[i, j] != samples[i, j - 1]
                push!(histogram_times, times[j] - time_prev)
                time_prev = times[j]
                push!(histogram_heights, abs(samples[i, j] - samples[i, j - 1]))
            end
        end
    end

    return histogram_heights, histogram_times
end

end # module BNPAnalysis
