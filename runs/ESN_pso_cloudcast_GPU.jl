include("../ESN.jl")
using Metaheuristics


############################################################################ DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")

# files   = ["2017M01.nc","2017M02.nc","2017M03.nc","2017M04.nc","2017M05.nc" ]
# all     = cat([ ncread(dir*fl, "__xarray_dataarray_variable__") for fl in files ]... , dims=1)

# u = cc_to_int(all[1,:,:])
# countmap(u)
# Images.Gray.(u./10)



############################################################################ PARAMS
sd = rand(1:10000)
Random.seed!(sd)
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:wb_logger_name    => "ESN_pso_cloudcast_GPU"
    ,:layers            => [(1,1000)]
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

if _params[:gpu] CUDA.allowscalar(false) end

par = filter(kv-> kv[1] ∉ [:wb_logger_name,:train_f,:test_f,:lg], _params)
if _params[:wb]
    using Logging, Wandb
    _params[:lg] = wandb_logger(_params[:wb_logger_name])
    Wandb.log(_params[:lg], par )
end
display(par)



############################################################################ TARGET DATA
_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , step            = _params[:step]
)



############################################################################ PSO ALGORITHM
function fitness(x)
    A,D,R = x[1], x[2], x[3]

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [ [1.0  for _ in 1:layer[1]] for layer in _params[:layers]]
        ,:Rin_dens => [ [1.0  for _ in 1:layer[1]] for layer in _params[:layers]]
        ,:sigma    => [ [1.0  for _ in 1:layer[1]] for layer in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:layer[1]] for layer in _params[:layers]]
        ,:alpha    => [ [A    for _ in 1:layer[1]] for layer in _params[:layers]]
        ,:density  => [ [D    for _ in 1:layer[1]] for layer in _params[:layers]]
        ,:rho      => [ [R    for _ in 1:layer[1]] for layer in _params[:layers]]
    )

    r1=[]
    tm = @elapsed begin
        r1 = do_batch_dwesn(_params_esn,_params, sd)
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", r1.error, "\n", printime  )

    return r1.error
end

pso_dict = Dict(
    "N"  => 20
    ,"C1" => 2.0
    ,"C2" => 1.5
    ,"w"  => 1.0
    ,"max_iter" => 50
)


for _it in 1:1
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        if _it == 1
            Wandb.log(_params[:lg], pso_dict )
        end
        Wandb.log(_params[:lg], par )
    else
        display(par)
        display(pso_dict)
        println(" ")
    end

    pso = PSO(;information=Metaheuristics.Information()
        ,N  = pso_dict["N"]
        ,C1 = pso_dict["C1"]
        ,C2 = pso_dict["C2"]
        ,ω  = pso_dict["w"]
        ,options = Options(iterations=pso_dict["max_iter"])
    )

    # Cota superior e inferior de individuos. alpha, density, rho
    lx = [0.01, 0.01, 0.01 ]'
    ux = [0.99,  0.7, 4.0  ]'
    lx_ux = vcat(lx,ux)

    res = optimize( fitness, lx_ux, pso )

    if _params[:wb]
        close(_params[:lg])
    end
end


# EOF 
