Base.@kwdef mutable struct ESN
    id        ::Int     = 0
    R         ::Mtx     = zeros(1,1)
    R_in      ::Mtx     = zeros(1,1)
    R_fdb     ::Mtx     = zeros(1,1)
    R_out     ::Mtx     = zeros(1,1)
    Y         ::Mtx     = zeros(1,1)
    X         ::Mtx     = zeros(1,1)
    x         ::Mtx     = zeros(1,1)
    R_size    ::Int16   = size(R,1)
    R_scaling ::Float64 = 1.0
    alpha     ::Float64 = 0.5
    beta      ::Float64 = 1.0e-8
    rho       ::Float64 = 1.0
    sigma     ::Float64 = 1.0
    sgmd      ::Function= tanh
    F_in      ::Function= (f,u) -> R_in * f(u)
    input_active ::Bool = true
    output_active::Bool = true
end


Base.@kwdef mutable struct layerESN
    esns            ::Vector{ESN}
    nodes           ::Int16         = sum([e.R_size for e in esns])
end

Base.@kwdef mutable struct DWESN
    layers          ::Vector{layerESN}
    input_to_all    ::Bool          = false
    train_function  ::Function      = __do_train_DWESN_cloudcast!
    test_function   ::Function      = __do_test_DWESN_cloudcast_pixel!
    X               ::Mtx           = zeros(1,1)
    R_out           ::Mtx           = zeros(1,1)
    beta            ::Float64       = 1.0e-8
    wrong_class     ::Any           = []
    classes_Y       ::Any           = []
    Y_target        ::Any           = []
    Y               ::Any           = []
    error           ::Dict{Any,Any} = Dict()
    classes_Routs   ::Dict{Int16,Dict{Int16,Union{Array{Float64},CuArray}}} = Dict()
    esns            ::Dict{Int,Any} = Dict(_esn.id => _esn for l in layers for _esn in l.esns)
    connections     ::Dict{Int,Any} = Dict()
    output_size     ::Int           = sum([_e.R_size for layer in layers for _e in layer.esns if _e.output_active])
end
