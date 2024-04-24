
function split_data_cloudcast(;data, train_length, test_length, target_pixel, radius, step=1)

    d,tp,rd,trl,tel = data,target_pixel, radius, train_length, test_length

    train_x   = cc_to_int(d[1:trl         , tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd ])
    test_x    = cc_to_int(d[trl+1:trl+tel , tp[1]-rd:tp[1]+rd , tp[2]-rd:tp[2]+rd ])

    if typeof(step) == Int
        train_y = cc_to_int(d[1+step:trl+step         , tp[1] , tp[2]])
        test_y  = cc_to_int(d[trl+1+step:trl+tel+step , tp[1] , tp[2]])
    else
        train_y = [cc_to_int(d[1+step[i]:trl+step[i]        , tp[1], tp[2]]) for i in step]
        test_y  = [cc_to_int(d[trl+1+step[i]:trl+tel+step[i], tp[1], tp[2]]) for i in step]
    end
    
    return train_x, train_y, test_x, test_y

end