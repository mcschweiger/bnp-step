module Distribs

using LinearAlgebra
using Random
using Distributions

"""
    MultivariateGaussian.sample(mean::Vector{Float32}, sigma::Matrix{Float32}, epsilon::Float32=1f-8)

Samples from a multivariate Gaussian distribution.

# Arguments
- `mean::Vector{Float32}`: Mean vector of the distribution.
- `sigma::Matrix{Float32}`: Covariance matrix of the distribution.
- `epsilon::Float32`: Small value added to the diagonal of the covariance matrix for numerical stability. Default is `1f-8`.

# Returns
A vector sampled from the multivariate Gaussian distribution.
"""
struct MultivariateGaussian
    mean::Vector{Float32}
    sigma::Matrix{Float32}
    epsilon::Float32

    function MultivariateGaussian(mean::Vector{Float32}, sigma::Matrix{Float32}, epsilon::Float32=1f-8)
        new(mean, sigma + epsilon * I)
    end
end

function sample(dist::MultivariateGaussian)
    L = cholesky(dist.sigma).L
    z = randn(length(dist.mean))
    return dist.mean .+ L * z
end

end # module Distribs
