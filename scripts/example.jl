# include("../install.jl")
using Pkg
Pkg.activate(".")
Pkg.resolve()
# Pkg.activate("/home/max/codes/stepfind/bnp-step/")
# includet("/home/max/codes/stepfind/bnp-step/src/BNPStep.jl")
using BNPStep 
using StatsBase

# Simulate N_data datapoints, using BNPStep.BNPAnalysis.kernel() as-defined for the step kernel
# To simulate or analyze a different kernel, redefine BNPStep.BNPAnalysis.kernel() in ./src/BNPAnalysis.jl
N_data=1000
dataset, truth = simulate_and_save_ground_truth("/tmp/synthetic_data_"*string(N_data)*".h5"; N=N_data)

# Instantiate the sampler using truth
step_model = BNP_Step_from_ground_truth(truth)

# Run BNP-Step with N_samples iterations
N_samples = 500
@time results = analyze(step_model, dataset, N_samples)

# Write the results to H5
# save_results("./results/test.h5", results)


# Generate plots, with 95% CIs, median, MAP and last sample
fig = visualize_results(results, dataset; plot_type="step")
display(fig)