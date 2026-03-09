using Pkg
homepath = ARGS[2]# = "/home/max/""/home/max/"#
bnpdir = joinpath(homepath,"codes/stepfind/", endswith(homepath,"max/") ? "BNPStep/" : "bnp-step")
Pkg.activate(bnpdir)
Pkg.instantiate()
include(joinpath(bnpdir,"src/BNPStep.jl"))
using .BNPStep   # assuming visualize_results is part of this module
using StatsBase

j = parse(Int,ARGS[1])
j=1
tracepaths = readdir(joinpath(homepath,"codes/stepfind/washu-stl-traces"),join=true)
data_paths = [readdir(tracepaths[j],join=true) for j in eachindex(tracepaths)]
for j in eachindex(data_paths)
    data_paths[j] = data_paths[j][endswith.(data_paths[j],".txt")]
end
n_iters_segment = 10
n_segments = 500

for data_path in data_paths[j]
    # data_path = "/home/max/codes/stepfind/washu-stl-traces/5mer_data_traces"
    # j = 1
    for seg in 1:n_segments
        @show data_path

        
        dataset = load_data_txt(joinpath(data_path,"Trk150_10_ns_drift_Qub.txt"), true)

        step_model = BNP_Step_(B_max = 150)
        model_initialized = true

        # results = analyze(step_model, dataset, 1)

        if model_initialized

        results = analyze(step_model, dataset, n_iters_segment)
        model_initialized = false

        else
        # outpath = data_path[1:end-4]*"-$seg-results.h5"


        results_snapshot = emit_results_snapshot(results)
        step_model = BNP_Step_(B_max=150,truth = Dict{String,Any}(results_snapshot), init_temperature=1,load_initialization = :ground_truth)
        model_initialized = true
        results = analyze(step_model, dataset, n_iters_segment)
        model_initialized = false
        end

        outpath = data_path[1:end-4]*"-$seg-results.h5"
        save_results(outpath,results)

    end
end
# data_path = "/home/max/codes/stepfind/washu-stl-traces/10mer_data_traces/Trk171_ns_13_drift_Qub.txt"
# outpath = data_path[1:end-4]*"-2-results.h5"
#         dataset = load_data_txt(data_path, true)

# # results_snapshot = Dict{String,Vector}(BNPStep.emit_results_snapshot(Dict{String,Any}(results)))
# results = BNPStep.load_results(outpath)
if false
        outpath = data_path*"/Trk150_10_ns_drift_Qub-10-results.h5"
        results = BNPStep.load_results(outpath)
        segs = 20:10:170
        for seg in segs
            outpath = data_path*"/Trk150_10_ns_drift_Qub-$seg-results.h5"
            results_seg = BNPStep.load_results(outpath)
            for key in keys(results_seg)
                results[key] = vcat(results[key],results_seg[key])
            end
        end

fig = BNPStep.visualize_results(Dict{String,Vector}(results), dataset; plot_type="step", font_size = 32)

keys(results)


display(fig)
t_n = dataset["times"]
signal_n = []
for s in eachindex(results["posterior"])
push!(signal_n, BNPStep.reconstruct_signal_from_sample(
           results["b_m"][s],
           results["h_m"][s],
           results["t_m"][s],
           results["dt"][s],
           results["f_bg"][s],
           kernel,
           t_n
       ))
end


dsignal = diff.(signal_n)
dsignal_flag = [dsig.>0 for dsig in dsignal]
befores = []
afters = []
for (flags, signal) in zip(dsignal_flag, signal_n)
    for ind in findall(flags)
        push!(befores,signal[ind])
        push!(afters,signal[ind+1])
    end

end 
hist2
end
# visualize_results(results, dataset; plot_type="hist_step_height")
# visualize_results(results, dataset; plot_type="hist_dwell_time")
# visualize_results(results, dataset; plot_type="hist_emission")
# visualize_results(results, dataset; plot_type="hist_height_separated")
# visualize_results(results, dataset; plot_type="hist_f")
# visualize_results(results, dataset; plot_type="hist_eta")
# visualize_results(results, dataset; plot_type="survivorship")
# visualize_results(results, dataset; plot_type="hist_dwell_separated")
# visualize_results(results, dataset; plot_type="hist_emission_separated")
# using GLMakie
# timing_results = Dict{String, Vector{Float64}}()
# for N = [100000, 50000, 10000, 5000, 1000]

#     dataset, truth = simulate_and_save_ground_truth("/tmp/synthetic_data_"*string(N)*".h5"; N)
#     # truth["posterior"] = results["posterior"]
#     # Run BNP-Step with minimal iterations
#     step_model = BNP_Step_from_ground_truth(truth)
#     t = Vector{Float64}(undef, 10)
#     for j in eachindex(t)
#         results, t[j], bytes, gctime, memallocs= @timed analyze(step_model, dataset, 100)
#     end

#     setindex!(timing_results, t, string(N))
# end


# python_results = [[19.49234914779663, 19.456262350082397, 19.5950448513031, 19.421010494232178, 19.87716555595398, 19.830528020858765, 19.645302057266235, 19.69860005378723, 19.405774116516113, 20.164286375045776],
# [117.50456547737122, 119.92050504684448, 123.25828146934509, 123.77206325531006, 119.83926939964294, 114.92958545684814, 116.8296377658844, 117.71984934806824, 116.78962206840515, 118.3127429485321],
# [283.6859722137451, 284.00522351264954, 285.91174840927124, 283.4481928348541, 283.9282627105713, 283.3654429912567, 290.7908718585968, 287.1783595085144, 290.6998734474182, 282.3999080657959],
# [1963.1806108951569, 1964.227620124817, 1978.2177095413208, 1952.74884390831, 1923.7392387390137, 1962.1503641605377, 1944.8201005458832, 1948.9032514095306, 1928.152910232544, 1934.9238712787628],
# [3848.689832687378]]

# fig = Figure()
# ax  = Axis(fig[1,1])
# plot!(ax, log10.(parse.(Float64, keys(timing_results))), log10.(mean.(values(timing_results))); label = "Julia")
# plot!(ax, log10.([1000, 5000, 10000, 50000, 100000]), log10.(mean.(python_results)), label = "Python")

# axislegend(ax; position = (:right, :bottom))
# ax.xlabel = "log10(#Datapoints)"
# ax.ylabel = "log10(wall time per 100 iterations [seconds])"
# display(fig)        outpath = data_path[1:end-4]*"-$seg-results.h5"
# results = BNPStep.load_results(outpath)