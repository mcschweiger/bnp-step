using Pkg
Pkg.activate("/home/max/codes/stepfind/BNPStep/")
include("/home/max/codes/stepfind/BNPStep/src/BNPStep.jl")
using .BNPStep   # assuming visualize_results is part of this module
# using GLMakie
using StatsBase


# N_data is the number of simulated datapoints
N_data=1000

# Create the synthetic dataset and write it to a temporary location
dataset, truth = simulate_and_save_ground_truth("/tmp/synthetic_data_"*string(N_data)*".h5"; N = N_data)

# Use the output "truth" dictionary to instantiate the sampler context
step_model = BNP_Step_from_ground_truth(truth)

# Run 
results = analyze(step_model, dataset, 10)
save_results("./results/test.h5", results)
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
timing_results = Dict{String, Vector{Float64}}()
for N = [100000, 50000, 10000, 5000, 1000]

    dataset, truth = simulate_and_save_ground_truth("/tmp/synthetic_data_"*string(N)*".h5"; N)
    # truth["posterior"] = results["posterior"]
    # Run BNP-Step with minimal iterations
    step_model = BNP_Step_from_ground_truth(truth)
    t = Vector{Float64}(undef, 10)
    for j in eachindex(t)
        results, t[j], bytes, gctime, memallocs= @timed analyze(step_model, dataset, 100)
    end

    setindex!(timing_results, t, string(N))
end


python_results = [[19.49234914779663, 19.456262350082397, 19.5950448513031, 19.421010494232178, 19.87716555595398, 19.830528020858765, 19.645302057266235, 19.69860005378723, 19.405774116516113, 20.164286375045776],
[117.50456547737122, 119.92050504684448, 123.25828146934509, 123.77206325531006, 119.83926939964294, 114.92958545684814, 116.8296377658844, 117.71984934806824, 116.78962206840515, 118.3127429485321],
[283.6859722137451, 284.00522351264954, 285.91174840927124, 283.4481928348541, 283.9282627105713, 283.3654429912567, 290.7908718585968, 287.1783595085144, 290.6998734474182, 282.3999080657959],
[1963.1806108951569, 1964.227620124817, 1978.2177095413208, 1952.74884390831, 1923.7392387390137, 1962.1503641605377, 1944.8201005458832, 1948.9032514095306, 1928.152910232544, 1934.9238712787628],
[3848.689832687378]]

fig = Figure()
ax  = Axis(fig[1,1])
plot!(ax, log10.(parse.(Float64, keys(timing_results))), log10.(mean.(values(timing_results))); label = "Julia")
plot!(ax, log10.([1000, 5000, 10000, 50000, 100000]), log10.(mean.(python_results)), label = "Python")

axislegend(ax; position = (:right, :bottom))
ax.xlabel = "log10(#Datapoints)"
ax.ylabel = "log10(wall time per 100 iterations [seconds])"
display(fig)