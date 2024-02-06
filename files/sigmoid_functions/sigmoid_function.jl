function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end


# x = [x/10 for x in -50:50]
# y = [sigmoid(i) for i in x ]
# plot(x,y; title="sigmoid" )