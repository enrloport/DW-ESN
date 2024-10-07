
function split_data_cloudcast(;data, train_length, test_length, target_pixel, radius, steps=[1], train_start=0)

    d,tp,rd,trl,tel,trs = data, target_pixel, radius, train_length, test_length, train_start

    train_x   = cc_to_int(d[trs+1:trs+trl         , tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd ])
    test_x    = cc_to_int(d[trs+1+trl:trs+trl+tel , tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd ])

    train_y = Dict(s => cc_to_int(d[trs+1+s:trs+trl+s        , tp[1], tp[2]]) for s in steps)
    test_y  = Dict(s => cc_to_int(d[trs+1+trl+s:trs+trl+tel+s, tp[1], tp[2]]) for s in steps)
    
    return train_x, train_y, test_x, test_y
end