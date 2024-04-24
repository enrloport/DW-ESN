function _step_mnist(deepE, data,t,f)
    for _esn in deepE.layers[1].esns
        a = data[:,:,t]
        __update(_esn, a, f )
    end

    for i in 2:length(deepE.layers)
        for _esn in deepE.layers[i].esns
            v = vcat([_e.x for _e in deepE.layers[i-1].esns ]...)
            __update(_esn, v , f )
        end
    end
end

function _step_mnist_dwesnia(deepE, data,t,f)
    a = data[:,:,t]
    for _esn in deepE.layers[1].esns
        __update(_esn, a, f )
    end

    for i in 2:length(deepE.layers)
        for _esn in deepE.layers[i].esns
            v = vcat(f(a),[_e.x for _e in deepE.layers[i-1].esns ]...)
            __update(_esn, v , f )
        end
    end
end