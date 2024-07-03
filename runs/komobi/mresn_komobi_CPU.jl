include("../../ESN.jl")
using DelimitedFiles

using HDF5, H5Zblosc, H5Zbzip2, H5Zlz4, H5Zzstd

# DATASET
dir     = "data/komobi/"
file    = "accidentes20G_2023_12_18_2.h5"


_d = h5open(dir*file, "r")
# data = CSV.read(dir*file, DataFrame, header=false)


data = read(_d)

data["data"]
data["time_windows"]
data["index"]



# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => false
    ,:wb                => false
    ,:confusion_matrix  => true
    ,:wb_logger_name    => "MRESN_cloudcast_pixel_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 50000
    ,:test_length       => 1000
    ,:train_f           => __do_train_DWESN_cloudcast!
    ,:test_f            => __do_test_DWESN_cloudcast_pixel!
    ,:target_pixel      => (30,30)
    ,:radius            => 3
    ,:step              => 1
    ,:data              => all
)