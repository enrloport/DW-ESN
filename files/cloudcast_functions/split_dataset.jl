
function split_data_cloudcast(;data, train_length, test_length, target_pixel, radius, steps=[1])

    d,tp,rd,trl,tel = data,target_pixel, radius, train_length, test_length

    train_x   = cc_to_int(d[1:trl         , tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd ])
    test_x    = cc_to_int(d[trl+1:trl+tel , tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd ])

    train_y = Dict(s => cc_to_int(d[1+s:trl+s        , tp[1], tp[2]]) for s in steps)
    test_y  = Dict(s => cc_to_int(d[trl+1+s:trl+tel+s, tp[1], tp[2]]) for s in steps)
    
    return train_x, train_y, test_x, test_y
end