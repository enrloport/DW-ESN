include("ESN.jl")

using CSV, DataFrames, ImageShow


# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")


_a = cc_to_int(all)

# maximum(_a)
# ImageShow.gif(_a./10)

imgs = _a ./10

using ImageView
imshow(imgs, axes=(2,3))






