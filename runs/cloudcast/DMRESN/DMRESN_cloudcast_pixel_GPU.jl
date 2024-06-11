include("../../../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")

layers = [ [200,300], [100,100], ones(3).*50 ]
num_layers = length(layers)
res_per_layer = [length(l) for l in layers] 

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => true
    ,:input_to_all      => false
    ,:wb_logger_name    => "DMRESN_cloudcast_pixel_H1to4-100_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 4900
    ,:test_length       => 100
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_pixel!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:data              => all
)
_params[:input_size] = ((_params[:radius]*2)+1)^2

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    )
    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


for _ in 1:repit
    dwE=[]
    _params[:layers] = [(200,500), (200,300), (50,100)]
    connections = [[]]
    sd = rand(1:10000)
    Random.seed!(sd)
    # _params[:layers] = [(2,300)]; sd=776; Random.seed!(sd) # error 0.2875

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),length(layer) ) for layer in _params[:layers] ]
        ,:density  => [rand(Uniform(0.1,0.3),length(layer) ) for layer in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.1,0.5),length(layer) ) for layer in _params[:layers]]
        ,:rho      => [rand(Uniform(1.0,4.0),length(layer) ) for layer in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:length(_params[:layers][i])] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> sum(x), _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Target pixel"        => _params[:target_pixel]
        , "Radius"              => _params[:radius]
        , "Initial transient"   => _params[:initial_transient]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        )
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    end
    display(par)

    tm = @elapsed begin
        dwE = do_batch_dwesn(_params_esn,_params)
    end
    _params[:total_time] = tm
    full_log(_params,_params_esn,dwE)

    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", dwE.error, "\n", printime  )

end

# EOF