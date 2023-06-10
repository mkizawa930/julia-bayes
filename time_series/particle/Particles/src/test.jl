include("./Particles.jl")

using Distributions
using Test

function random_walk(n, σ=1.0)
    (σ * randn(n)) |> cumsum
end

@testset "test" begin
    f = x -> begin
        x
    end
    h = x -> begin
        x
    end
    σw = 0.1
    σv = 10.0
    pf = Particles.ParticleFilter(
        n_particles = 100,
        f = f,
        h = h,
        state_dims = [1],
        sys_noise_dist = Normal(0, σw),
        obs_noise_dist = Normal(0, σv)
    )
    
    ParticleFilters.fit(pf, )

end