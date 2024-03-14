
function split_data_cloudcast(;data, train_length, test_length, target_pixel, radius, step=1)

    d,tp,rd,trl,tel = data,target_pixel, radius, train_length, test_length

    train_x = d[1:trl                   , tp[1]-rd:tp[1]+rd    , tp[2]-rd:tp[2]+rd ]
    train_y = d[1+step:trl+step         , tp[1]                , tp[2]]

    test_x  = d[trl+1:trl+tel           , tp[1]-rd:tp[1]+rd    , tp[2]-rd:tp[2]+rd ]
    test_y  = d[trl+1+step:trl+tel+step , tp[1]                , tp[2]]

    
    return cc_to_int(train_x), cc_to_int(train_y), cc_to_int(test_x), cc_to_int(test_y)

end