module BNPSampler

using LinearAlgebra
using Random
using Distributions
using ..BNPAnalysis
import SpecialFunctions: loggamma

"""
    sample_b(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, gamma, rng, temp, kernel)

Samples all loads `b_m` as part of BNP-Step's Gibbs sampler.

# Arguments
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `h_m_vec`, `t_m_vec`: Previous samples.
- `f_vec`, `eta_vec`, `nu_vec`: Previous parameters.
- `gamma`: Hyperparameter for priors on `b_m`.
- `rng`: Random number generator.
- `temp`: Temperature (for simulated annealing).
- `kernel`: Kernel function.

# Returns
A vector containing the new `b_m` sample.
"""
function sample_b(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, gamma, rng, temp, kernel)
    times_matrix = repeat(t_m_vec[end]', num_data, 1)
    obs_time_matrix = repeat(data_times, 1, weak_limit)
    height_matrix = repeat(h_m_vec[end]', num_data, 1)

    eta_nu = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)

    height_heaviside_mat = isempty(dt_vec) ?
        height_matrix .* kernel(obs_time_matrix .- times_matrix) :
        height_matrix .* kernel(obs_time_matrix .- times_matrix, dt_vec[end])

    load_matrix = repeat(b_m_vec[end]', num_data, 1)
    gumbel_variables = -log.(-log.(rand(rng, weak_limit, 2)))
    sampling_order = shuffle(rng, 1:weak_limit)

    for i in sampling_order
        load_matrix[:, i] .= 1
        bht_matrix = load_matrix .* height_heaviside_mat
        bht_sum = f_vec[end] .+ sum(bht_matrix, dims=2)
        exponent_on = sum(-0.5 .* eta_nu .* (data_points .- bht_sum).^2)

        load_matrix[:, i] .= 0
        bht_matrix = load_matrix .* height_heaviside_mat
        bht_sum = f_vec[end] .+ sum(bht_matrix, dims=2)
        exponent_off = sum(-0.5 .* eta_nu .* (data_points .- bht_sum).^2)

        prob_on = (1 / temp) * log(gamma / weak_limit) + (1 / temp) * exponent_on
        prob_off = (1 / temp) * log(1 - (gamma / weak_limit)) + (1 / temp) * exponent_off

        new_bm = argmax([prob_off + gumbel_variables[i, 1], prob_on + gumbel_variables[i, 2]])
        load_matrix[:, i] .= new_bm - 1
    end

    return load_matrix[1, :]
end

"""
    sample_fh(weak_limit, num_data, data_points, data_times, b_m_vec, t_m_vec, dt_vec, eta_vec, nu_vec, psi, chi, f_ref, h_ref, rng, temp, kernel)

Samples `F_bg` and all `h_m` as part of BNP-Step's Gibbs sampler.

# Arguments
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `t_m_vec`: Previous samples.
- `f_vec`, `eta_vec`, `nu_vec`: Previous parameters.
- `psi`, `chi`: Variance hyperparameters.
- `f_ref`, `h_ref`: Mean hyperparameters.
- `rng`: Random number generator.
- `temp`: Temperature (for simulated annealing).
- `kernel`: Kernel function.

# Returns
A vector containing the new `F_bg` sample as the first element, followed by the new `h_m` samples.
"""
function sample_fh(
    weak_limit::Int,
    num_data::Int,
    data_points::Vector{Float32},
    data_times::Vector{Float32},
    b_m_vec::Vector{<:AbstractVector{Bool}},
    t_m_vec::Vector{Vector{Float32}},
    dt_vec::Vector{Float32},
    eta_vec::Vector{Float32},
    nu_vec::Vector{Float32},
    psi::Float32,
    chi::Float32,
    f_ref::Float32,
    h_ref::Float32,
    rng::AbstractRNG,
    temp::Float32,
    kernel::Function
)::Vector{Float32}
    b_m = b_m_vec[end]
    t_m = t_m_vec[end]
    dt = dt_vec[end]
    η = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)

    on_idx = findall(b_m)
    off_idx = findall(!, b_m)

    if isempty(on_idx)
        # No active steps: sample f and h_i independently from prior
        fh = zeros(Float32, weak_limit + 1)
        fh[1] = rand(rng, Normal(f_ref, sqrt(1 / psi)))
        fh[2:end] .= rand.(rng, Normal(h_ref, sqrt(1 / chi)), weak_limit)
        return fh
    end

    # === Design matrix: each column is a kernel(t - t_m[i]) for active i
    H = zeros(Float32, num_data, length(on_idx))
    for (j, i) in enumerate(on_idx)
        H[:, j] .= kernel(data_times .- t_m[i], dt)
    end

    # === Full design matrix: prepend column of 1s for background
    X = hcat(ones(Float32, num_data), H)

    # === Precision and posterior
    Λ = Diagonal(η)
    prior_prec = Diagonal(vcat(psi, fill(chi, length(on_idx))))
    posterior_prec = X' * Λ * X + prior_prec
    rhs = X' * (Λ * data_points) + prior_prec * vcat(f_ref, fill(h_ref, length(on_idx)))

    # === Sample from Gaussian posterior
    cov = temp .* inv((posterior_prec + posterior_prec') / 2)
    mean = cov * rhs
    cov = (cov + cov') / 2  # ensure symmetry

    # Optional stabilization
    eigmin = minimum(eigvals(cov))
    if eigmin < 1f-8
        cov .+= (1f-8 - eigmin + 1f-10) * I
    end

    sample = rand(rng, MvNormal(mean, cov))

    # === Assemble full vector of [f; h1, h2, ..., hB]
    fh = zeros(Float32, weak_limit + 1)
    fh[1] = sample[1]
    for (j, i) in enumerate(on_idx)
        fh[i+1] = sample[j+1]
    end
    fh[2:end][off_idx] .= rand.(rng, Normal(h_ref, sqrt(1 / chi)))  # from prior

    return fh
end


"""
    sample_t(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, rng, temp, kernel)

Samples all `t_m` as part of BNP-Step's Gibbs sampler.

# Arguments
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `h_m_vec`, `t_m_vec`: Previous samples.
- `f_vec`, `eta_vec`, `nu_vec`: Previous parameters.
- `rng`: Random number generator.
- `temp`: Temperature (for simulated annealing).
- `kernel`: Kernel function.

# Returns
A vector containing the new `t_m` sample.
"""
function sample_t(
    weak_limit::Int,
    num_data::Int,
    data_points::Vector{Float32},
    data_times::Vector{Float32},
    b_m_vec::Vector{<:AbstractVector{Bool}},
    h_m_vec::Vector{Vector{Float32}},
    t_m_vec::Vector{Vector{Float32}},
    dt_vec::Vector{Float32},
    f_vec::Vector{Float32},
    eta_vec::Vector{Float32},
    nu_vec::Vector{Float32},
    rng::AbstractRNG,
    temp::Float32,
    kernel::Function
)::Vector{Float32}

    b_m = b_m_vec[end]
    h_m = h_m_vec[end]
    t_m = copy(t_m_vec[end])  # mutable copy
    dt = dt_vec[end]
    f = f_vec[end]
    η = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)

    sampling_order = shuffle(rng, eachindex(b_m))
    u_values = rand(rng, length(b_m))

    for i in sampling_order
        if !b_m[i]
            # @show "skipping load $i: $b_m[i]"
            t_m[i] = rand(rng, data_times)
            continue
        end

        t_old = t_m[i]
        # t_prop = rand(rng, data_times)
        t_prop = clamp(t_old + rand(rng, [-1, 0, 1]),extrema(data_times)...)
        if t_prop == t_old
            continue
        end

        ll_old = calculate_loglikelihood(
            weak_limit, num_data, data_points, data_times,
            [b_m], [h_m], [t_m], [dt], [f], [η], nu_vec, kernel
        )

        # Compute log likelihood for proposed t_i
        t_m[i] = t_prop
        ll_prop = calculate_loglikelihood(
            weak_limit, num_data, data_points, data_times,
            [b_m], [h_m], [t_m], [dt], [f], [η], nu_vec, kernel
        )

        # Metropolis-Hastings
        # @show  [ll_old , ll_prop , temp]
        log_alpha = (ll_prop - ll_old) / temp
        if log(u_values[i]) >= log_alpha
            t_m[i] = t_old
        end
    end

    return t_m
end

"""
    sample_eta_metropolis(eta_vec, nu_vec, weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec,
                          phi, eta_ref, rng, temp, kernel)

Samples `eta` as part of BNP-Step's Gibbs sampler using the Metropolis-Hastings algorithm.

# Arguments
- `eta_vec`: Previous `eta` samples.
- `nu_vec`: Observation variances.
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `h_m_vec`, `t_m_vec`: Previous samples.
- `dt_vec`, `f_vec`: Previous parameters.
- `phi`: Shape hyperparameter for Gamma prior on `eta`.
- `eta_ref`: Scale hyperparameter for Gamma prior on `eta`.
- `rng`: Random number generator.
- `temp`: Temperature (for simulated annealing).
- `kernel`: Kernel function.

# Returns
The new `eta` sample.
"""
function sample_eta_metropolis(eta_vec, nu_vec, weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec,
                                phi, eta_ref, rng, temp, kernel)
    new_eta = eta_vec[end]
    proposal_eta = new_eta + 1 * rand(rng , Normal(0, 1))  # Propose a new `eta`

    if proposal_eta > 0
        # Calculate log-likelihood for the proposed `eta`
        log_likelihood_proposal = calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                                          t_m_vec, dt_vec, f_vec, [proposal_eta], nu_vec, kernel)
        log_prior_proposal = (phi - 1) * log(proposal_eta) - (phi * proposal_eta) / eta_ref

        # Calculate log-likelihood for the current `eta`
        log_likelihood_current = calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                                         t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, kernel)
        log_prior_current = (phi - 1) * log(new_eta) - (phi * new_eta) / eta_ref

        # Calculate acceptance ratio in log space
        log_acceptance_ratio = (log_likelihood_proposal + log_prior_proposal) - (log_likelihood_current + log_prior_current)

        # Accept or reject the proposal
        if log(rand(rng)) < log_acceptance_ratio
            new_eta = proposal_eta
        end
    end

    return new_eta
end

"""
    sample_dt_metropolis(eta_vec, nu_vec, weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec,
                         chi, dt_ref, rng, temp, kernel)

Samples `dt` as part of BNP-Step's Gibbs sampler using the Metropolis-Hastings algorithm.

# Arguments
- `eta_vec`: Previous `eta` samples.
- `nu_vec`: Observation variances.
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `h_m_vec`, `t_m_vec`: Previous samples.
- `dt_vec`, `f_vec`: Previous parameters.
- `chi`: Shape hyperparameter for Gamma prior on `dt`.
- `dt_ref`: Scale hyperparameter for Gamma prior on `dt`.
- `rng`: Random number generator.
- `temp`: Temperature (for simulated annealing).
- `kernel`: Kernel function.

# Returns
The new `dt` sample.
"""
function sample_dt_metropolis(eta_vec, nu_vec, weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec,
                               chi, dt_ref, rng, temp, kernel)
    current_dt = dt_vec[end]
    proposal_dt = current_dt + Float32(rand(rng, Normal(0, 1)))  # Propose a new `dt`

    if proposal_dt > 0
        # Calculate log-likelihood for the proposed `dt`
        log_likelihood_proposal = calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                                          t_m_vec, [proposal_dt], f_vec, eta_vec, nu_vec, kernel)
        log_prior_proposal = (chi - 1) * log(proposal_dt) - (chi * proposal_dt) / dt_ref - loggamma(chi)

        # Calculate log-likelihood for the current `dt`
        log_likelihood_current = calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                                         t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, kernel)
        log_prior_current = (chi - 1) * log(current_dt) - (chi * current_dt) / dt_ref - loggamma(chi)

        # Calculate acceptance ratio in log space
        log_acceptance_ratio = (log_likelihood_proposal + log_prior_proposal) - (log_likelihood_current + log_prior_current)

        # Accept or reject the proposal
        if log(rand(rng)) < log_acceptance_ratio
            return proposal_dt
        end
    end

    return current_dt
end

"""
    calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                            t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, kernel)

Calculates the log likelihood given a dataset and a set of associated samples from BNP-Step.

# Arguments
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `h_m_vec`, `t_m_vec`: Previous samples.
- `dt_vec`, `f_vec`: Previous parameters.
- `eta_vec`, `nu_vec`: Observation variances.
- `kernel`: Kernel function.

# Returns
The log likelihood for the provided samples and observations.
"""
function calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
    t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, kernel)
# Determine eta_nu
eta_nu = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)

# Reconstruct full signal from the last sample
trace = reconstruct_signal_from_sample(
b_m_vec[end],
h_m_vec[end],
t_m_vec[end],
dt_vec[end],
f_vec[end],
kernel,
data_times[:]
)

# Compute log-likelihood
residual = data_points .- trace
exponent_term = sum((eta_nu ./ 2) .* residual.^2)
log_likelihood = sum(log.(eta_nu ./ (2π)) ./ 2) - exponent_term

return log_likelihood
end
"""
    calculate_logposterior(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                           t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, chi, h_ref, gamma, phi, eta_ref, psi, dt_ref, f_ref, kernel)

Calculates the log posterior given a dataset and a set of associated samples from BNP-Step.

# Arguments
- `weak_limit`: Maximum number of possible steps.
- `num_data`: Number of observations.
- `data_points`: Observations.
- `data_times`: Time points corresponding to each observation.
- `b_m_vec`, `h_m_vec`, `t_m_vec`: Previous samples.
- `dt_vec`, `f_vec`: Previous parameters.
- `eta_vec`, `nu_vec`: Observation variances.
- `chi`: Variance hyperparameter for priors on `h_m`.
- `h_ref`: Mean hyperparameter for priors on `h_m`.
- `gamma`: Hyperparameter for priors on `b_m`.
- `phi`: Shape hyperparameter for Gamma prior on `eta`.
- `eta_ref`: Scale hyperparameter for Gamma prior on `eta`.
- `psi`: Variance hyperparameter for prior on `F_bg`.
- `dt_ref`: Scale hyperparameter for Gamma prior on `dt`.
- `f_ref`: Mean hyperparameter for prior on `F_bg`.
- `kernel`: Kernel function.

# Returns
The log posterior and log likelihood for the provided samples and observations.
"""


function calculate_logposterior(weak_limit, num_data, data_points, data_times,
                                 b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec,
                                 eta_vec, nu_vec,
                                 chi, h_ref, gamma, phi, eta_ref, psi, dt_ref, f_ref,
                                 kernel)

    # Likelihood
    log_likelihood = calculate_loglikelihood(weak_limit, num_data, data_points, data_times,
                                             b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, kernel)

    # === Priors ===
    b_m = b_m_vec[end]
    h_m = h_m_vec[end]
    t_m = t_m_vec[end]
    dt = dt_vec[end]
    f = f_vec[end]
    eta = isempty(eta_vec) ? nothing : eta_vec[end]

    # Bernoulli-Beta process prior on b_m (approximate with Bernoulli log-prob)
    p_on = gamma / weak_limit
    prior_b_m = sum(log.(ifelse.(b_m .== 1, p_on, 1 - p_on)))

    # Normal prior on h_m
    prior_h_m = sum(logpdf.(Normal(h_ref, sqrt(1 / chi)), h_m))

    # Uniform prior on t_m
    prior_t_m = -weak_limit * log(num_data)

    # Gamma prior on η (if present)
    prior_eta = eta === nothing ? 0.0 :
                logpdf(Gamma(phi, eta_ref / phi), eta)

    # Gamma prior on dt
    prior_dt = logpdf(Gamma(chi, dt_ref / chi), dt)

    # Normal prior on f
    prior_f = logpdf(Normal(f_ref, sqrt(1 / psi)), f)

    # Combine all
    log_posterior = log_likelihood + prior_b_m + prior_h_m + prior_t_m + prior_eta + prior_dt + prior_f

    return log_posterior, log_likelihood
end

export sample_b, sample_fh, sample_t, sample_eta_metropolis, sample_dt_metropolis, calculate_loglikelihood, calculate_logposterior
end # module BNPSampler