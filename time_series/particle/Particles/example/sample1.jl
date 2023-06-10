include("../src/Particles.jl")

using .Particles
using Distributions
using DataStructures
using StatsBase
using Plots, StatsPlots

function random_walk(n, σ)
    x = randn(n) * σ
    return cumsum(x)
end


# サンプルデータ生成
y = random_walk(100, 1.0)

n_particles = 10000
f = x -> begin
    x
end
h = x -> begin
    x
end

# モデル定義
pf = Particles.ParticleFilter(
    n_particles=n_particles, 
    n_particle_dim=1,
    n_obs_dim=1,
    lags=20,
    f=f, 
    h=h, 
    init_particle_dists=Dict(:state => Normal(0, 2.0)),
    sys_noise_dist=Cauchy(0, 0.5),
    obs_noise_dist=Normal(0, sqrt(0.0001))
)

Particles.fit(pf, y)

# 実行
begin
    results = Particles.filter(pf, y)
    yhat = mean(stack(results.particle[:predicted][:]), dims=2) |> vec

    qs = quantile.(results.particle[:predicted], Ref([0.05, 0.95]))
    qs = hcat(qs...)
    lower = qs[1,:]
    upper = qs[2,:]
    p = plot(vec(y))
    p = plot!(p, 1:length(y), lower, fillrange=upper, 
        linewidth=0.0, 
        fillalpha=0.3, 
        c=1, # color palette
        label="y(95% region)"
    )
    title!("Random walk w ~ $(pf.sys_noise_dist) v ~ $(pf.obs_noise_dist)", titlefont=8)
    display(p)
end