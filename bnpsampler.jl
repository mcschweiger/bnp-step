module BNPSampler

using LinearAlgebra
using Random
using Distributions

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
function sample_fh(weak_limit, num_data, data_points, data_times, b_m_vec, t_m_vec, dt_vec, eta_vec, nu_vec, psi, chi, f_ref, h_ref, rng, temp, kernel)
    on_loads = findall(x -> x == 1, b_m_vec[end])
    off_loads = findall(x -> x == 0, b_m_vec[end])
    on_load_times = t_m_vec[end][on_loads]

    times_matrix = repeat(on_load_times', num_data, 1)
    obs_time_matrix = repeat(data_times, 1, length(on_loads))
    load_matrix = ones(num_data, length(on_loads))

    eta_nu = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)
    eta_nu = reshape(eta_nu, :, 1)

    load_heaviside_mat = isempty(dt_vec) ?
        load_matrix .* kernel(obs_time_matrix .- times_matrix) :
        load_matrix .* kernel(obs_time_matrix .- times_matrix, dt_vec[end])

    precision_matrix = load_heaviside_mat' * (eta_nu .* load_heaviside_mat) + chi * I(length(on_loads))
    precision_first_row = sum(eta_nu .* load_heaviside_mat, dims=1)
    precision_matrix = vcat(hcat(sum(eta_nu) + psi, precision_first_row), hcat(precision_first_row', precision_matrix))

    q_matrix = (data_points' * (eta_nu .* load_heaviside_mat)) + (chi * h_ref)
    q_matrix = vcat(sum(data_points .* eta_nu) + (psi * f_ref), q_matrix)

    covariance = temp * inv(precision_matrix)
    mean_vector = covariance * q_matrix

    h_tmp = rand(MvNormal(mean_vector, covariance))
    new_fh = zeros(weak_limit + 1)
    ind = 1

    for i in 1:(weak_limit + 1)
        if i in off_loads
            new_fh[i] = rand(Normal(h_ref, sqrt(1 / chi)))
        else
            new_fh[i] = h_tmp[ind]
            ind += 1
        end
    end

    return new_fh
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
function sample_t(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, rng, temp, kernel)
    # Pre-calculate matrices
    load_matrix = repeat(b_m_vec[end]', num_data, 1)
    obs_time_matrix = repeat(data_times, 1, weak_limit)
    height_matrix = repeat(h_m_vec[end]', num_data, 1)

    eta_nu = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)

    bh_matrix = load_matrix .* height_matrix
    times_matrix = repeat(t_m_vec[end]', num_data, 1)

    # Shuffle sampling order
    sampling_order = shuffle(rng, 1:weak_limit)

    # Pre-generate random values for Metropolis step
    u_values = rand(rng, weak_limit)

    for i in sampling_order
        if b_m_vec[end][i] == 0
            # Sample from the prior if the load is off
            times_matrix[:, i] .= rand(rng, data_times)
        else
            # Generate a proposal
            t_prop = rand(rng, data_times)
            t_old = times_matrix[1, i]

            # Calculate exponent for t_old
            bht_matrix = isempty(dt_vec) ?
                bh_matrix .* kernel(obs_time_matrix .- times_matrix) :
                bh_matrix .* kernel(obs_time_matrix .- times_matrix, dt_vec[end])
            bht_sum = f_vec[end] .+ sum(bht_matrix, dims=2)
            exponent_old = sum(-0.5 .* eta_nu .* (data_points .- bht_sum).^2)

            # Calculate exponent for t_prop
            times_matrix[:, i] .= t_prop
            bht_matrix = isempty(dt_vec) ?
                bh_matrix .* kernel(obs_time_matrix .- times_matrix) :
                bh_matrix .* kernel(obs_time_matrix .- times_matrix, dt_vec[end])
            bht_sum = f_vec[end] .+ sum(bht_matrix, dims=2)
            exponent_prop = sum(-0.5 .* eta_nu .* (data_points .- bht_sum).^2)

            # Calculate acceptance ratio in log space
            log_acceptance_ratio = (exponent_old - exponent_prop) / temp

            # Accept or reject the proposal
            if log(u_values[i]) >= log_acceptance_ratio
                times_matrix[:, i] .= t_old
            end
        end
    end

    return times_matrix[1, :]
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
    proposal_eta = new_eta + 1 * rand(Normal(0, 1, rng))  # Propose a new `eta`

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
    proposal_dt = current_dt + rand(Normal(0, 1, rng))  # Propose a new `dt`

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
    # Pre-calculate matrices
    times_matrix = repeat(t_m_vec[end]', num_data, 1)
    obs_time_matrix = repeat(data_times, 1, weak_limit)
    height_matrix = repeat(h_m_vec[end]', num_data, 1)
    load_matrix = repeat(b_m_vec[end]', num_data, 1)

    eta_nu = isempty(eta_vec) ? nu_vec : (eta_vec[end] .* nu_vec) ./ (eta_vec[end] .+ nu_vec)

    # Calculate product of b_m and h_m term-wise
    bh_matrix = load_matrix .* height_matrix

    # Calculate Heaviside terms times loads and heights
    bht_matrix = isempty(dt_vec) ?
        bh_matrix .* kernel(obs_time_matrix .- times_matrix) :
        bh_matrix .* kernel(obs_time_matrix .- times_matrix, dt_vec[end])

    # Calculate sum term for the exponent
    bht_sum = f_vec[end] .+ sum(bht_matrix, dims=2)
    exponent_term = sum((eta_nu ./ 2) .* (data_points .- bht_sum).^2)

    # Calculate log likelihood
    log_likelihood = sum(log.(eta_nu ./ (2 * π)) ./ 2) - exponent_term

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
function calculate_logposterior(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                 t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, chi, h_ref, gamma, phi, eta_ref, psi, dt_ref, f_ref, kernel)
    # Calculate the log likelihood
    log_likelihood = calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                             t_m_vec, dt_vec, f_vec, eta_vec, nu_vec, kernel)

    # Calculate priors on b_m
    b_m = b_m_vec[end]
    on_prior = (gamma / weak_limit) .* b_m
    off_prior = (1 .- b_m) .* (1 .- (gamma / weak_limit))
    prior_b_m = sum(log.(on_prior .+ off_prior))

    # Calculate priors on h_m
    h_m = h_m_vec[end]
    prior_h_m = ((weak_limit / 2) * log(chi / (2 * π))) - ((chi / 2) * sum((h_m .- h_ref).^2))

    # Calculate priors on t_m
    prior_t_m = -weak_limit * log(num_data)

    # Calculate prior on eta
    prior_eta = isempty(eta_vec) ? 0 : ((phi - 1) * log(eta_vec[end])) - ((phi * eta_vec[end]) / eta_ref) -
                                       loggamma(phi) - (phi * log(eta_ref / phi))

    # Calculate prior on dt
    prior_dt = ((chi - 1) * log(dt_vec[end])) - ((chi * dt_vec[end]) / dt_ref) - loggamma(chi) -
               (chi * log(dt_ref / chi))

    # Calculate prior on F
    prior_f = (0.5 * log(psi / (2 * π))) - ((psi / 2) * ((f_vec[end] - f_ref)^2))

    # Calculate the log posterior
    log_posterior = log_likelihood + prior_b_m + prior_h_m + prior_t_m + prior_eta + prior_f + prior_dt

    return log_posterior, log_likelihood
end

end # module BNPSampler