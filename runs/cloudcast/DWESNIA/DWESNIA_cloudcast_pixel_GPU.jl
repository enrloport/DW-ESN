include("../../../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")


# PARAMS
repit = 100
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "DWESN_cloudcast_pixel_H1to4-100_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 49000
    ,:test_length       => 1000
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


_w = 2
for _l in [2,3,4,5]

    for _ in 1:repit
        dwE=[]
        _params[:layers] = [ [_w,300] for _ in 1:_l ]
        _params[:connections] = Dict(
            (x+1) => [((x-_w+1),1.0)]
            for x in _w:(_l*_w)-1
        )
        _params[:active_inputs] = 1:_w*_l
        _params[:active_outputs]= (_l-1)*_w+1:_l*_w


        sd = rand(1:10000)
        Random.seed!(sd)
        # _params[:layers] = [(2,300)]; sd=776; Random.seed!(sd) # error 0.2875

        _params_esn = Dict{Symbol,Any}(
            :R_scaling => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
            ,:alpha    => [rand(Uniform(0.3,0.7),length(layer) ) for layer in _params[:layers]]
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
            , "Active inputs"       => _params[:active_inputs] 
            , "Active outputs"      => _params[:active_outputs]
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

        for l in 1:length(dwE.layers)
            for r in dwE.layers[l].esns
                println("Layer: ",l, ", id: ", r.id)
            end
        end

    end
end


# EOF