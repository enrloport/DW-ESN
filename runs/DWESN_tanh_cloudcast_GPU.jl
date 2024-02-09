include("../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
_data_o = ncread(dir*file, "__xarray_dataarray_variable__")

# u = cc_to_int(_data_o[1,:,:])
# countmap(u)
# Images.Gray.(u./10)


# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => false
    ,:wb                => false
    ,:wb_logger_name    => "DWESN_tanh_cloudcast_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 2
    ,:train_length      => 5000
    ,:test_length       => 200
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:step              => 1
)

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = _data_o
    , train_length    = _params[:train_length]
    , test_length     = _params[:train_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , step            = _params[:step]
    )


if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


include("../ESN.jl")
for _ in 1:repit
    _params[:layers] = [(1,200),(1,1000)]
    sd = rand(1:10000)
    Random.seed!(sd)
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),num_e[1] ) for num_e in _params[:layers] ]
        ,:density  => [rand(Uniform(0.01,0.2),num_e[1] ) for num_e in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.01,0.5),num_e[1] ) for num_e in _params[:layers]]
        ,:rho      => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:_params[:layers][i][1]] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> x[1]*x[2], _params[:layers] ) )
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

    r1=[]
    tm = @elapsed begin
        r1 = do_batch_dwesn(_params_esn,_params, sd)
    end
    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", r1.error, "\n", printime  )

end

# EOF