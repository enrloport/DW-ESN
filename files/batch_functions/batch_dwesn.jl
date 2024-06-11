
function do_batch_dwesn(_params_esn, _params)

    p,pe  = _params, _params_esn
    im_sz = p[:input_size]

    layers = []
    layer1 = layerESN( esns = [
                ESN( 
                     R      = new_R(p[:layers][1][i], density=pe[:density][1][i], rho=pe[:rho][1][i], gpu=p[:gpu])
                    ,R_in   = new_R_in(p[:layers][1][i], im_sz , sigma = pe[:sigma][1][i] ,gpu=p[:gpu], density=pe[:Rin_dens][1][i])
                    ,R_scaling = pe[:R_scaling][1][i], alpha = pe[:alpha][1][i], rho = pe[:rho][1][i], sigma = pe[:sigma][1][i], sgmd = pe[:sgmds][1][i]
                ) for i in 1:length(p[:layers][1])
            ])
    push!(layers,layer1)

    for l in 2:length(p[:layers])
        input_sz = p[:input_to_all] ? layers[l-1].nodes + im_sz : layers[l-1].nodes
        layer = layerESN( esns = [
            ESN(
                 R      = new_R(p[:layers][l][i], density=pe[:density][l][i], rho=pe[:rho][l][i], gpu=p[:gpu])
                ,R_in   = new_R_in(p[:layers][l][i], input_sz, sigma = pe[:sigma][l][i] ,gpu=p[:gpu], density=pe[:Rin_dens][l][i] )
                ,R_scaling = pe[:R_scaling][l][i], alpha  = pe[:alpha][l][i], rho = pe[:rho][l][i], sigma = pe[:sigma][l][i], sgmd = pe[:sgmds][l][i]
            ) for i in 1:length(p[:layers][l])
        ])
        push!(layers,layer)
    end   

    dwE = DWESN(
        layers = layers
        ,beta=p[:beta] 
        ,train_function = p[:train_f]
        ,test_function  = p[:test_f]
        ,input_to_all   = p[:input_to_all]
        )
    tm_train = @elapsed begin
        dwE.train_function(dwE,p)
    end
    tm_test = @elapsed begin
        dwE.test_function(dwE,p)
    end
    p[:train_time],p[:test_time] = tm_train, tm_test

    return dwE
end
