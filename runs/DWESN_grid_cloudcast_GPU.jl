include("../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")

# files   = ["2017M01.nc","2017M02.nc","2017M03.nc","2017M04.nc","2017M05.nc" ]
# all     = cat([ ncread(dir*fl, "__xarray_dataarray_variable__") for fl in files ]... , dims=1)

# u = cc_to_int(_data_o[1,:,:])
# countmap(u)
# Images.Gray.(u./10)

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:wb_logger_name    => "DWESN_grid_cloudcast_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 50000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:step              => 1
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

for N in [50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000]
    for A in [x/10 for x in 1:10]
        for D in [x/10 for x in 1:7]
            for R in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5]

                sd = rand(1:10000)
                Random.seed!(sd)

                _params[:layers] = [(1,N)]
                _params_esn = Dict{Symbol,Any}(
                    :R_scaling => [ [1.0 for _ in 1:_params[:layers][i][1]] for i in _params[:layers]]
                    ,:Rin_dens => [ [1.0 for _ in 1:_params[:layers][i][1]] for i in _params[:layers]]
                    ,:sigma    => [ [1.0 for _ in 1:_params[:layers][i][1]] for i in _params[:layers]]
                    ,:sgmds    => [ [tanh for _ in 1:_params[:layers][i][1]] for i in 1:length(_params[:layers]) ]
                    ,:alpha    => [ [A for _ in 1:_params[:layers][i][1]] for i in _params[:layers]]
                    ,:density  => [ [D for _ in 1:_params[:layers][i][1]] for i in _params[:layers]]
                    ,:rho      => [ [R for _ in 1:_params[:layers][i][1]] for i in _params[:layers]]
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
        end
    end
end


# EOF 
