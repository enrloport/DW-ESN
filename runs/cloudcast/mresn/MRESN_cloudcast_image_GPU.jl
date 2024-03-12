include("../../../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "DWESN_tanh_cloudcast_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 100
    ,:train_length      => 5000
    ,:test_length       => 1
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_image!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:step              => 1
    ,:data              => all
)

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , step            = _params[:step]
    )


# u = cc_to_int(_params[:train_data][1,:,:])
# u2 = cc_to_int(_params[:train_data][2,:,:])
# Images.Gray.(u./10)
    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


r1=[]
for _ in 1:repit
    _params[:layers] = [(4,300)]
    sd = rand(1:10000)
    Random.seed!(sd)
    # _params[:layers] = [(2,300)]; sd=776; Random.seed!(sd) # error 0.2875

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),num_e[1] ) for num_e in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),num_e[1] ) for num_e in _params[:layers] ]
        ,:density  => [rand(Uniform(0.01,0.2),num_e[1]) for num_e in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.01,0.5),num_e[1]) for num_e in _params[:layers]]
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

    tm = @elapsed begin
        r1 = do_batch_dwesn(_params_esn,_params)
    end
    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", r1.error, "\n", printime  )

end

r1.Y

_t = _params[:train_length] + _params[:test_length]
target = cc_to_int(all[_t,:,:])

# using DelimitedFiles
# writedlm( "mresn.csv",  r1.Y[1,:,:], ',')

# using CSV, DataFrames
# r1Y = Matrix(CSV.read("mresn_1000_50000_1.csv", DataFrame, header=false))


Images.Gray.(hcat(target, r1.Y[1,:,:])./10)
# EOF