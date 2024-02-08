# Load external libraries

using CUDA
using DataFrames
using Dates
using Distributions
using Images
using LinearAlgebra
using NetCDF
using Plots
using Random
using SparseArrays
using StableRNGs
using StatsBase

# Usefull libraries for specific experiments

# using Augmentor
# using BenchmarkTools
# using CSV
# using DelimitedFiles
# using GraphPlot
# using Graphs
# using MatrixDepot
# using Metaheuristics
# using MLDatasets
# using SimpleWeightedGraphs
# using Suppressor

# Types definition
Mtx = Union{Matrix,SparseMatrixCSC, Array, CuArray, Diagonal{Bool, Vector{Bool}}, Diagonal{Bool, CuArray{Bool, 1, CUDA.Mem.DeviceBuffer}}  }
Data= Union{DataFrame, Mtx, Vector, Array, CuArray}

# Includes
include.( filter(contains(r".jl$"), readdir("./files/"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/batch_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/preprocess_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/reservoir_generators"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/sigmoid_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/test_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/train_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/update_functions/"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/wandb_functions"; join=true)))
