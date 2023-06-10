include("../src/Particles.jl")

using MarketData

ts = yahoo(Symbol("^N225"))

y = values(ts[:AdjClose][end-500:end]) |> skipmissing |> collect

# 状態関数
f = x -> begin x
    F = [2.0 -1.0; 1.0 0.0]
    return F * x
end

# 観測関数
h = x -> begin x
    H = [1.0, 0.0]
    return H' * x
end

# モデル
model = Particles.ParticleFilter(
    n_particles = 10000,
    n_particle_dim = 2,
    n_obs_dim = 1,
    f = f,
    h = h,
    init_particle_dists = Dict(:state => Normal(mean(y), std(y))),
    sys_noise_dist = Normal(0, 100),
    obs_noise_dist = TDist(1.0)
)

# pars = Particles.init_particles(model)
# pars = Particles.update(model, pars)
# pred_pars = Particles.observe(model, pars)
# factory = Particles.DistributionFactory(model.obs_noise_dist)
# weights = Particles.calc_weights!(weights, y[1], pred_pars, factory)
# pars = Particles.resample(pars, weights)
# Particles.update(model, pars)

results = Particles.fit(model, y)
yhat = mean.(results.particle[:smoothed])
plot(y, size=(800,600)); plot!(yhat)