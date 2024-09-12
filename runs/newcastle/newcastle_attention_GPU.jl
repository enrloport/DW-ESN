include("../../ESN.jl")
using Metaheuristics

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"

file2= "newcastle/newcastle_cloud.nc"

_info     = ncinfo(dir*file)

_imgs   = ncread(dir*file, "__xarray_dataarray_variable__")#[:,30-3:30+3,30-3:30+3]
_windD  = ncgetatt(dir*file2, "global", "Wind Direction")
_hum    = ncgetatt(dir*file2, "global", "Humidity")
_windS  = ncgetatt(dir*file2, "global", "Wind Speed")
_press  = ncgetatt(dir*file2, "global", "Pressure")

maxi = length(_hum)

_imgs, _hum, _windS, _windD, _press = _imgs[1:maxi,:,:], _hum[1:maxi]./10.0, _windS[1:maxi]./100.0, _windD[1:maxi]./360.0, _press[1:maxi]./1000

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "newcastle_attention_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 100
    ,:train_length      => 4900
    ,:test_length       => 100
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_pixel!
    ,:target_pixel      => (25,50)
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:train_data        => Dict()
    ,:test_data         => Dict()
)

function split_aditional_data(train_length, test_length, var)
    tr,te   = train_length, test_length
    train_x = var[1:tr        , : ]
    test_x  = var[tr+1:tr+te  , : ]

    return train_x, test_x
end



_params[:train_data][1],  _params[:train_labels],  _params[:test_data][1],  _params[:test_labels] = split_data_cloudcast(
    data              = _imgs
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    )

_params[:train_data][2],  _params[:test_data][2] = split_aditional_data( _params[:train_length], _params[:test_length], _press)
_params[:train_data][3],  _params[:test_data][3] = split_aditional_data( _params[:train_length], _params[:test_length], _windS)


_params[:input_size] = ((_params[:radius]*2)+1)^2
_params[:additional_inputs_size] = [size(_params[:train_data][i],2) for i in 2:length(keys(_params[:train_data]))]

    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


# for _ in 1:repit

    dwE=[]
    _params[:layers] = [ [200 for _ in 1:5],[300,300]]
    _params[:connections] = Dict(
         6 => [(i,1.0) for i in 1:5 ]
        ,7 => [(i,1.0) for i in 1:5 ]
    )

    _params[:active_inputs]     = [1,2,3,4,5,6,7]
    _params[:active_outputs]    = [6,7]
    # _params[:attention_inputs]  = Dict( 
    #     4 => [1]
    #     ,5 => [2]
    # )

    sd = 42#rand(1:10000)
    Random.seed!(sd)

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),num_e[1] ) for num_e in _params[:layers] ]
        ,:density  => [rand(Uniform(0.1,0.3),num_e[1] ) for num_e in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.1,0.5),num_e[1] ) for num_e in _params[:layers]]
        ,:rho      => [rand(Uniform(1.0,4.0),num_e[1] ) for num_e in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:_params[:layers][i][1]] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> sum(x), _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Target Pixel"        => _params[:target_pixel]
        , "Radius"              => _params[:radius]
        , "Initial transient"   => _params[:initial_transient]
        , "Active inputs"       => _params[:active_inputs]
        , "Active outputs"      => _params[:active_outputs]
        , "Layers"              => _params[:layers]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        , "rho"                 => _params_esn[:rho][1][1]
        , "sigma"               => _params_esn[:sigma][1][1]
        , "reservoirs"          => sum([x[1] for x in _params[:layers]])
        , "nodes"               => sum( [ sum(l) for l in _params[:layers] ] )
        , "alpha min"           => minimum( vcat( _params_esn[:alpha]...) )
        , "alpha max"           => maximum( vcat( _params_esn[:alpha]...) )
        , "density min"         => minimum( vcat( _params_esn[:density]...) )
        , "density max"         => maximum( vcat( _params_esn[:density]...) )
    )




    include("../../ESN.jl")
    tm = @elapsed begin
        dwE = do_batch_dwesn_attention(_params_esn,_params)
    end

    err_dict = Dict("Error_step_"*string(s) => dwE.error[s] for s in _params[:steps] )

    if _params[:wb]
       _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], merge(par, err_dict) )
    end
    display(par)

    _params[:total_time] = tm
    full_log(_params,_params_esn,dwE)

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", dwE.error, "\n", printime  )

    if _params[:wb]
        close(_params[:lg])
    end

# end


# EOF