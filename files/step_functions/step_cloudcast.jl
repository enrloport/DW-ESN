function get_input(_esn,u,f, extra)
    res = _esn.input_active ? reshape(u, :, 1) : zeros(0)
    for a in keys(_esn.additional_inputs)
        res = vcat(res,extra[a])
    end
    return f(res)
end

function _step_cloudcast(dwE, data,f; extra_inputs = Dict())
    for layer in dwE.layers
        for _esn in layer.esns
            inpt = get_input(_esn,data,f,extra_inputs )
            conns = _esn.id in keys(dwE.connections) ? [cn[1].x .* cn[2] for cn in dwE.connections[_esn.id] if cn[2] != 0 ] : []
            v = vcat(conns..., inpt)
            __update(_esn, v , f )
        end
    end    
end