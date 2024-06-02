module EGG

using CSV
using DataFrames
using DelimitedFiles
using Statistics
using StatsBase
using LinearAlgebra
using Random

#auxilary
export parseargs

#generator
export wGCL
export wGCL_directed

# Include package code
include("auxilary.jl")
include("generator.jl")
end