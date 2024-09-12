function get_input(_esn,u,f)
    res = _esn.input_active ? u : zeros(0)
    for a in _esn.additional_inputs
        res = hcat(res,a)
    end
    return f(res)
end

function _step_cloudcast(dwE, data,t,f)
    u = data[t,:,:]
    # a = 
    for layer in dwE.layers
        for _esn in layer.esns
            # inpt = _esn.input_active ? f(u) : f(zeros(0))
            inpt = get_input(_esn,u,f)
            conns = _esn.id in keys(dwE.connections) ? [cn[1].x .* cn[2] for cn in dwE.connections[_esn.id] if cn[2] != 0 ] : []
            v = vcat(conns..., inpt)
            __update(_esn, v , f )
        end
    end    
end