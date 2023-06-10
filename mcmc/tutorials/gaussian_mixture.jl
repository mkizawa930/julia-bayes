using Distributions
using FillArrays
using StatsPlots

using LinearAlgebra
using Random; Random.seed!(3);

w = [0.5, 0.5] # weights
μ = [-3.5, 0.5] # means

mixturemodel = MixtureModel([MvNormal(Fill(μ_k, 2), I) for μ_k in μ], w)

N = 60
x = rand(mixturemodel, N)

scatter(x[1,:], x[2,:], legend=false)

using Turing
@model function gaussian_mixture_model(x)
    K = 2 # components
    
    # 初期平均値
    μ ~ MvNormal(Zeros(K), I)

    # 各カテゴリの生起確率
    w ~ Dirichlet(K, 1.0)
    distribution_assignments = Categorical(w)

    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μk, D), I) for μk in μ]

    k = Vector{Int}(undef, N) # 各時刻iのクラスタインデックス
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

model = gaussian_mixture_model(x)
sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ, :w))
n_samples = 100
n_chains = 3
chains = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
plot(chains[["μ[1]", "μ[2]"]]; colordim=:parameter, legend=true)