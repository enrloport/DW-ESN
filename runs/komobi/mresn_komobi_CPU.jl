include("../../ESN.jl")
using DelimitedFiles

# DATASET
dir     = "data/"
file    = "accidentes20G_2023_12_test_kike3.csv"

data = CSV.read(dir*file, DataFrame, header=false)

data


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