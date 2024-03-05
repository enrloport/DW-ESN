include("../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")

# files   = ["2017M01.nc","2017M02.nc","2017M03.nc","2017M04.nc","2017M05.nc" ]
# all     = cat([ ncread(dir*fl, "__xarray_dataarray_variable__") for fl in files ]... , dims=1)

# u = cc_to_int(all[1,:,:])
# countmap(u)
# Images.Gray.(u./10)

# PARAMS
sd = rand(1:10000)
Random.seed!(sd)
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "MRESN_gridAll_cloudcast_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 50000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:seed              => sd
    ,:step              => 1
)

par = filter(kv-> kv[1] ∉ [:wb_logger_name,:train_f,:test_f,:lg], _params)
par = Dict(
           string(kv[1]) => kv[2]
           for kv in par
)
if _params[:wb]
    using Logging, Wandb
    _params[:lg] = wandb_logger(_params[:wb_logger_name])
    Wandb.log(_params[:lg], par )
end
display(par)

if _params[:gpu] CUDA.allowscalar(false) end

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , step            = _params[:step]
    )

for r in [2,3,4]
    for R in [1.0, 2.0, 3.0, 4.0, 5.0]
        for A in [x/10 for x in 1:9]
            for D in [x/10 for x in 1:7]
                for N in [50, 100, 200, 300, 400, 500]

                    # sd = rand(1:10000)
                    Random.seed!(_params[:seed])

                    _params[:layers] = [(1,N)]
                    _params_esn = Dict{Symbol,Any}(
                        :R_scaling => [ [1.0 for _ in 1:layer[1]]   for layer in _params[:layers]]
                        ,:Rin_dens => [ [1.0 for _ in 1:layer[1]]   for layer in _params[:layers]]
                        ,:sigma    => [ [1.0 for _ in 1:layer[1]]   for layer in _params[:layers]]
                        ,:sgmds    => [ [tanh for _ in 1:layer[1]]  for layer in _params[:layers]]
                        ,:alpha    => [ [A for _ in 1:layer[1]]     for layer in _params[:layers]]
                        ,:density  => [ [D for _ in 1:layer[1]]     for layer in _params[:layers]]
                        ,:rho      => [ [R for _ in 1:layer[1]]     for layer in _params[:layers]]
                    )

                    r1=[]
                    tm = @elapsed begin
                        r1 = do_batch_dwesn(_params_esn,_params)
                    end

                    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
                    println("Error: ", r1.error, "\n", printime  )
                end
            end
        end
    end
end

if _params[:wb]
    close(_params[:lg])
end

# EOF 
