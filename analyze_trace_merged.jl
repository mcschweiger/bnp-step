using BNPStep
using BNPInputs
using BNPStep
using DelimitedFiles
using JLD2

# Load data from CSV file
function load_csv_data(filepath::String; has_timepoints::Bool=false)
    if has_timepoints
        data = readdlm(filepath, ',', Float64)
        times = data[:, 1]
        observations = data[:, 2]
    else
        observations = readdlm(filepath, ',', Float64)
        times = nothing
    end
    return Dict("data" => observations, "times" => times)
end

# Main script
function main()
    # Filepath to the dataset
    filepath = "./yovan_data/trace_merged.csv"

    # Load the dataset
    println("Loading data from: $filepath")
    data = load_csv_data(filepath, has_timepoints=true)

    # Initialize BNPStep object
    step = BNPStep(
        chi=0.028,
        dt_ref=100.0,
        h_ref=10.0,
        psi=0.0028,
        F_ref=0.0,
        phi=1.0,
        eta_ref=10.0,
        gamma=1.0,
        B_max=50,
        load_initialization="prior",
        use_annealing=true,
        init_temperature=2250,
        scale_factor=1.25,
        seed=42
    )

    # Analyze the data
    println("Analyzing data...")
    num_samples = 100000
    results = analyze(step, data, num_samples)

    # Save results to a file
    output_file = "./yovan_data/trace_merged_results.jld2"
    println("Saving results to: $output_file")
    @save output_file results

    # Visualize results
    println("Visualizing results...")
    visualize_results(results, data, plot_type="step")
    visualize_results(results, data, plot_type="hist_step_height")
end

main()
