using Pkg
Pkg.activate(".")
includet("./src/BNPStep.jl")
using .BNPStep   # assuming visualize_results is part of this module
using GLMakie


dataset, truth = simulate_and_save_ground_truth("/tmp/synthetic_data.h5")
# truth["posterior"] = results["posterior"]
# Run BNP-Step with minimal iterations
step_model = BNP_Step_from_ground_truth(truth)
@profview results = analyze(step_model, dataset, 500)

# Generate plots
fig = visualize_results(results, dataset; plot_type="step")
display(fig)
# visualize_results(results, dataset; plot_type="hist_step_height")
# visualize_results(results, dataset; plot_type="hist_dwell_time")
# visualize_results(results, dataset; plot_type="hist_emission")
# visualize_results(results, dataset; plot_type="hist_height_separated")
# visualize_results(results, dataset; plot_type="hist_f")
# visualize_results(results, dataset; plot_type="hist_eta")
# visualize_results(results, dataset; plot_type="survivorship")
# visualize_results(results, dataset; plot_type="hist_dwell_separated")
# visualize_results(results, dataset; plot_type="hist_emission_separated")
