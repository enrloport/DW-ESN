function _step_cloudcast(dwE, data,t,f)
    for _esn in dwE.layers[1].esns
        a = data[t,:,:]
        __update(_esn, a, f )
    end

    for i in 2:length(dwE.layers)
        for _esn in dwE.layers[i].esns
            v = vcat([_e.x for _e in dwE.layers[i-1].esns ]...)
            __update(_esn, v , f )
        end
    end
end