
function do_batch_dwesn(_params_esn, _params)

    p,pe          = _params, _params_esn
    l_r           = vcat(p[:layers]...)
    szl           = length(l_r)
    input_sz      = (id) -> id in p[:active_inputs] ? p[:input_size] : 0
    fsz           = (id) -> sum( [ l_r[i] for i in 1:szl if id in keys(p[:connections]) && i in getfield.(filter(x -> x[2] != 0, p[:connections][id]), 1) ] ) + input_sz(id)
    id_counter    = 0
    layers        = []

    for l in 1:length(p[:layers])
        num = l > 1 ? length(p[:layers][l-1]) : 0
        id_counter += num
        layer = layerESN( esns = [
            ESN( id     = id_counter + i
                ,R      = new_R(p[:layers][l][i], density=pe[:density][l][i], rho=pe[:rho][l][i], gpu=p[:gpu])
                ,R_in   = new_R_in(p[:layers][l][i], fsz(id_counter + i) , sigma = pe[:sigma][l][i] ,gpu=p[:gpu], density=pe[:Rin_dens][l][i] )
                ,R_scaling = pe[:R_scaling][l][i], alpha  = pe[:alpha][l][i], rho = pe[:rho][l][i], sigma = pe[:sigma][l][i], sgmd = pe[:sgmds][l][i]
                ,input_active = (id_counter + i) in p[:active_inputs]
                ,output_active= (id_counter + i) in p[:active_outputs]
            ) for i in 1:length(p[:layers][l])
        ])
        push!(layers,layer)

    end   

    dwE = DWESN(
        layers = layers
        ,beta=p[:beta] 
        ,train_function = p[:train_f]
        ,test_function  = p[:test_f]
        )
    dwE.connections = Dict(
        k => [(dwE.esns[cn[1]],cn[2]) for cn in p[:connections][k] ]
        for k in keys(p[:connections])
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
