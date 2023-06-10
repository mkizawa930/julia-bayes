using Distributed

addprocs(4) # マルチプロセス

@everywhere using Turing

@everywhere @model function gdemo(x)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))

    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s²))
    end
end

@everywhere model = gdemo([1.5, 2.0])

sample(model, NUTS(), MCMCDistributed(), 1000, 4)