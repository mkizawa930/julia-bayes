using Distributions

function LocationScaledModel(model::Type{T}, μ, args...) where T <: Distribution
    @assert !hasproperty(fieldnames(model), :μ)
    return LocationScale(μ, σ, model(args...))
end


function ShiftedTDist(μ, args...)
    LocationScale(μ, 1.0, TDist(args...))
end