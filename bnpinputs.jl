################################################################################
# BNP-Step Julia Transpilation: bnpinputs.jl
# Original Python code from: https://github.com/LabPresse/bnp-step/tree/bnp-step-loading-WIP
# Transpiled and GPU-accelerated where applicable by Max Schweiger, with contributions
# and adaptations by Github copilot, with funding from Arizona State University (ASU)
# and the National Institutes of Health (NIH).
#
# License: MIT
################################################################################

module BNPInputs

using CSV
using DataFrames

include("utils/clean_parse.jl")  # Split out helper functions for clarity

export load_data_garcia, load_data_txt, load_data_csv,
       load_data_HMM, load_data_expt, load_data_kv

"""
    load_data_garcia(filename::String, data_format::String; path::Union{Nothing, String}=nothing)

Loads a specific trace from a CSV file based on the given model type and iteration.

# Arguments
- `filename::String`: Name of the CSV file (without extension).
- `data_format::String`: A two-character string specifying the model type and iteration (e.g., "B1", "0R").
- `path::Union{Nothing, String}`: Path to the file. If `nothing`, assumes the file is in the current directory.

# Returns
A dictionary containing:
- `data::Vector{Float64}`: The extracted trace values.
- `times::Union{Vector{Float64}, Nothing}`: The corresponding time points.
- `nu_vec::Union{Vector{Float64}, Nothing}`: The associated nu values (if present).
- `ground_truths::Nothing`
- `parameters::Nothing`

# Throws
- `ArgumentError` if `data_format` is invalid or required columns are missing.
- `ArgumentError` if no rows match the specified `data_format`.
"""
function load_data_garcia(filename::String, data_format::String; path::Union{Nothing, String}=nothing)
    if !(typeof(filename) <: String)
        throw(ArgumentError("filename should be of type String, got $(typeof(filename))"))
    end
    if !(typeof(data_format) <: String) || length(data_format) != 2
        throw(ArgumentError("data_format must be a string of length 2 (e.g., 'B1', '0R')."))
    end

    full_path = isnothing(path) ? filename * ".csv" : joinpath(path, filename * ".csv")
    df = CSV.File(full_path, stringtype=String) |> DataFrame

    model_col = :model in names(df) ? :model : (:Model in names(df) ? :Model : nothing)
    iteration_col = :iteration in names(df) ? :iteration : (:Iteration in names(df) ? :Iteration : nothing)

    if isnothing(model_col) || isnothing(iteration_col)
        throw(ArgumentError("CSV must contain either 'model' or 'Model' and 'iteration' or 'Iteration' columns."))
    end

    subset = filter(row -> row[model_col] == data_format[1] && row[iteration_col] == data_format[2], df)

    if nrow(subset) == 0
        throw(ArgumentError("No rows match the specified data_format."))
    end

    dataset = Dict(
        "data" => subset[:, :data],
        "times" => :times in names(subset) ? subset[:, :times] : nothing,
        "nu_vec" => :nu_vec in names(subset) ? subset[:, :nu_vec] : nothing,
        "ground_truths" => nothing,
        "parameters" => nothing
    )
    return dataset
end



"""
    load_data_txt(filename::String, has_timepoints::Bool; path::Union{Nothing, String}=nothing)

Data loader for generic data sets in `.txt` format.

# Arguments
- `filename::String`: Name of the file to be loaded (without extension).
- `has_timepoints::Bool`: Whether the file has time points associated with observations.
- `path::Union{Nothing, String}`: Path to the file. If `nothing`, assumes the file is in the current directory.

# Returns
A dictionary containing:
- `data::Vector{Float64}`: The extracted data values.
- `times::Union{Vector{Float64}, Nothing}`: The corresponding time points (if present).
- `ground_truths::Nothing`
- `parameters::Nothing`
"""
function load_data_txt(filename::String, has_timepoints::Bool; path::Union{Nothing, String}=nothing)
    # Validate input and construct file path
    full_name = filename * ".txt"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    dataset = Dict{String, Any}()

    if has_timepoints
        data = Float64[]
        times = Float64[]
        # Read file line by line
        open(full_path, "r") do f
            for line in eachline(f)
                split_data = split(line, ",")
                push!(times, parse(Float64, split_data[1]))
                push!(data, parse(Float64, split_data[2]))
            end
        end
        dataset["data"] = data
        dataset["times"] = times
    else
        data = Float64[]
        # Read file line by line
        open(full_path, "r") do f
            for line in eachline(f)
                push!(data, parse(Float64, line))
            end
        end
        dataset["data"] = data
        dataset["times"] = nothing
    end

    dataset["ground_truths"] = nothing
    dataset["parameters"] = nothing

    return dataset
end



"""
    load_data_csv(filename::String, has_timepoints::Bool; path::Union{Nothing, String}=nothing)

Data loader for generic data sets in `.csv` format.

# Arguments
- `filename::String`: Name of the file to be loaded (without extension).
- `has_timepoints::Bool`: Whether the file has time points associated with observations.
- `path::Union{Nothing, String}`: Path to the file. If `nothing`, assumes the file is in the current directory.

# Returns
A dictionary containing:
- `data::Vector{Float64}`: The extracted data values.
- `times::Union{Vector{Float64}, Nothing}`: The corresponding time points (if present).
- `nu_vec::Union{Vector{Float64}, Nothing}`: The associated nu values (if present).
- `ground_truths::Nothing`
- `parameters::Nothing`
"""
function load_data_csv(filename::String, has_timepoints::Bool; path::Union{Nothing, String}=nothing)
    # Validate input and construct file path
    full_name = filename * ".csv"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    dataset = Dict{String, Any}()

    if has_timepoints
        # Read CSV file into a DataFrame
        df = CSV.File(full_path, header=false) |> DataFrame
        data_np = Matrix(df)

        # Handle case where first row contains strings
        if typeof(data_np[1, 1]) <: AbstractString
            df = CSV.File(full_path) |> DataFrame
            data_np = Matrix(df)
        end

        times = data_np[:, 1]
        data = data_np[:, 2]
        nu_vec = data_np[:, 3]

        # Build dictionary for output
        dataset["data"] = Float64.(data)
        dataset["times"] = Float64.(times)
        dataset["nu_vec"] = Float64.(nu_vec)
    else
        # Read CSV file into a DataFrame
        df = CSV.File(full_path, header=false) |> DataFrame
        data_np = Matrix(df)

        data = data_np[:, 1]
        nu_vec = data_np[:, 2]

        # Build dictionary for output
        dataset["data"] = Float64.(data)
        dataset["times"] = nothing
        dataset["nu_vec"] = Float64.(nu_vec)
    end

    dataset["ground_truths"] = nothing
    dataset["parameters"] = nothing

    return dataset
end


"""
    load_data_HMM(filename::String; path::Union{Nothing, String}=nothing)

Data loader for HMM-style data sets, as given in "An accurate probabilistic step finder for time-series analysis".

# Arguments
- `filename::String`: Name of the file to be loaded (without extension).
- `path::Union{Nothing, String}`: Path to the file. If `nothing`, assumes the file is in the current directory.

# Returns
A dictionary containing:
- `data::Vector{Float64}`: The extracted data values.
- `times::Vector{Float64}`: The corresponding time points.
- `ground_truths::Dict{String, Vector{Float64}}`: Ground truth trajectory data.
- `parameters::Dict{String, Any}`: Synthetic data generation parameters.
"""
function load_data_HMM(filename::String; path::Union{Nothing, String}=nothing)
    # Validate input and construct file path
    full_name = filename * ".csv"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    # Load all data from the CSV file
    df = CSV.File(full_path) |> DataFrame
    data_mat = Matrix(df)
    times = data_mat[:, 1]
    data = data_mat[:, 2]

    # Extract ground truth trajectory data
    ground = Dict(
        "x" => data_mat[:, 3],
        "u" => data_mat[:, 4]
    )

    # Extract synthetic data generation parameters
    params = Dict{String, Any}()
    params["type"] = "hmm"

    # Count the ground truth number of steps (transitions) in the data
    num_steps = sum(diff(ground["x"]) .!= 0)
    params["gt_steps"] = num_steps
    params["num_observations"] = length(ground["x"])
    params["f_back"] = parse(Float64, string(names(df)[3]))

    if length(unique(data_mat[:, 3])) > 2
        params["h_step"] = parse(Float64, string(names(df)[6]))
        params["h_step2"] = parse(Float64, string(names(df)[7]))
        params["h_step3"] = parse(Float64, string(names(df)[8]))
        params["h_step4"] = parse(Float64, string(names(df)[9]))
        params["h_step5"] = parse(Float64, string(names(df)[10]))
    else
        params["h_step"] = 0.0
        params["h_step2"] = parse(Float64, string(names(df)[6]))
        params["h_step3"] = nothing
        params["h_step4"] = nothing
        params["h_step5"] = nothing
    end
    params["eta"] = parse(Float64, string(names(df)[4]))

    # Pack everything into a dictionary
    dataset = Dict(
        "data" => data,
        "times" => times,
        "ground_truths" => ground,
        "parameters" => params
    )

    return dataset
end

"""
    load_data_expt(filename::String; path::Union{Nothing, String}=nothing)

Data loader for experimental data sets, as given in "An accurate probabilistic step finder for time-series analysis".

# Arguments
- `filename::String`: Name of the file to be loaded (without extension).
- `path::Union{Nothing, String}`: Path to the file. If `nothing`, assumes the file is in the current directory.

# Returns
A dictionary containing:
- `data::Vector{Float64}`: The extracted data values.
- `times::Vector{Float64}`: The corresponding time points.
- `ground_truths::Nothing`
- `parameters::Nothing`
"""
function load_data_expt(filename::String; path::Union{Nothing, String}=nothing)
    # Validate input and construct file path
    full_name = filename * ".txt"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    times = Float64[]
    data = Float64[]

    # Read in data from the file
    open(full_path, "r") do f
        for line in eachline(f)
            split_data = split(line, ",")
            push!(times, parse(Float64, split_data[1]))
            push!(data, parse(Float64, split_data[2]))
        end
    end

    # Build dictionary for output
    dataset = Dict(
        "data" => data,
        "times" => times,
        "ground_truths" => nothing,
        "parameters" => nothing
    )

    return dataset
end

"""
    load_data_kv(filename::String; path::Union{Nothing, String}=nothing)

Data loader for KV-type data sets, as given in "An accurate probabilistic step finder for time-series analysis".

# Arguments
- `filename::String`: Name of the file to be loaded (without extension).
- `path::Union{Nothing, String}`: Path to the file. If `nothing`, assumes the file is in the current directory.

# Returns
A dictionary containing:
- `data::Vector{Float64}`: The extracted data values.
- `times::Vector{Float64}`: The corresponding time points.
- `ground_truths::Dict{String, Vector{Float64}}`: Ground truth trajectory data.
- `parameters::Dict{String, Any}`: Synthetic data generation parameters.
"""
function load_data_kv(filename::String; path::Union{Nothing, String}=nothing)
    # Validate input and construct file path
    full_name = filename * ".txt"
    full_path = isnothing(path) ? full_name : joinpath(path, full_name)

    # Initialize variables
    ground_b = Float64[]
    ground_h = Float64[]
    ground_t = Float64[]
    data = Float64[]
    times = Float64[]

    # Read in file
    open(full_path, "r") do f
        # Read in ground truth parameters
        B_str = strip(readline(f))
        N_str = strip(readline(f))
        t_aqr_str = strip(readline(f))
        t_exp_str = strip(readline(f))
        F_str = strip(readline(f))
        h_stp_str = strip(readline(f))
        t_stp_str = strip(readline(f))
        eta_str = strip(readline(f))
        t_min_str = strip(readline(f))
        B_max_str = strip(readline(f))

        N_file = parse(Int, N_str)

        # Skip padding zeros
        for _ in 1:(N_file - 10)
            readline(f)
        end

        for i in 1:5
            for _ in 1:N_file
                value = parse(Float64, strip(readline(f)))
                if i == 1
                    push!(ground_b, value)
                elseif i == 2
                    push!(ground_h, value)
                elseif i == 3
                    push!(ground_t, value)
                elseif i == 4
                    push!(data, value)
                else
                    push!(times, value)
                end
            end
        end
    end

    # Extract synthetic data generation parameters
    B_file = parse(Int, B_str)
    F_file = parse(Float64, F_str)
    h_stp_file = parse(Float64, h_stp_str)
    t_stp_file = parse(Float64, t_stp_str)
    eta_file = parse(Float64, eta_str)
    B_max_file = parse(Int, B_max_str)

    params = Dict(
        "type" => "kv",
        "gt_steps" => B_file,
        "num_observations" => length(data),
        "f_back" => F_file,
        "h_step" => h_stp_file,
        "t_step" => t_stp_file,
        "eta" => eta_file,
        "B_max" => B_max_file
    )

    # Sanitize and pack ground truth trajectory data
    ground_b = ground_b[1:(B_max_file + 1)]
    ground_h = ground_h[1:(B_max_file + 1)]
    ground_t = ground_t[1:(B_max_file + 1)]

    ground = Dict(
        "b_m" => ground_b,
        "h_m" => ground_h,
        "t_m" => ground_t
    )

    # Pack everything into a dictionary
    dataset = Dict(
        "data" => data,
        "times" => times,
        "ground_truths" => ground,
        "parameters" => params
    )

    return dataset
end

end # module BNPInputs