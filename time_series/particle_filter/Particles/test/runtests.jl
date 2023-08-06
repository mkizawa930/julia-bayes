
using Particles
using Test

function random_walk(n, σ)
    x = randn(n) * σ
    return cumsum(x)
end

@testset "test DistributionFactory" begin
    factory = Particles.DistributionFactory(Normal, OrderedDict(:μ=>0.0, :σ=>1.0))
    Particles.getinstance(factory, μ=10.0)
end

@testset "test methods used in filter function" begin
    # サンプルデータ生成
    y = random_walk(100, 1.0) |> x -> reshape(x, 1, :)

    n_particles = 10000
    # 状態遷移関数
    f = x -> begin
        x
    end
    # 観測関数
    h = x -> begin
        x
    end
    pf = Particles.ParticleFilter(
        n_particles=n_particles, 
        n_state_dim=1,
        n_obs_dim=1,
        f=f, 
        h=h, 
        init_particle_dists=Dict(:state => Normal(0, 2.0)),
        sys_noise_model=Normal(0, 1.0),
        obs_noise_model=Normal(0, sqrt(0.0001))
    )

    # test init_particles
    pars = Particles.init_particles(pf)
    pred_pars = copy(pars)
    Particles.update(pf, pars)
    Particles.predict(pf, pars)

    # test calc_weights
    weights = zeros(pf.n_particles)
    factory = Particles.DistributionFactory(pf.obs_noise_dist)
    Particles.calc_weights!(weights, y[1], pars, factory)

    # test resampling
    pars = Particles.resample(pars, weights)
end

@testset "test filter" begin

    results = Particles.filter(pf, y)
end