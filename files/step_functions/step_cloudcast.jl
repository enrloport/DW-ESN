function _step_cloudcast(dwE, data,t,f)
    ia = dwE.input_to_all ? f(data[t,:,:]) : f(zeros(0))
    for _esn in dwE.layers[1].esns
        a = data[t,:,:]
        __update(_esn, a, f )
    end

    for i in 2:length(dwE.layers)
        for _esn in dwE.layers[i].esns
            v = vcat([_e.x for _e in dwE.layers[i-1].esns ]..., ia)
            __update(_esn, v , f )
        end
    end
end