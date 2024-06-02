using DelimitedFiles
using Random

fn_comms = "example/com300.dat"
fn_out = "example/edge300.attr"

function cat_col(com, comms)
    cat = rand() < 0.5 ? com : rand(comms)
    return "cat" * string(cat)
end

num_col(com) = randn() + com

comm = readdlm(fn_comms, Int)
comm_rows, no_cols = size(comm)

if no_cols == 2
    comm = comm[sortperm(comm[:, 1]), 2]
    comm = reshape(comm, size(comm)[1], 1)
end
comms = unique(comm)
Random.seed!(42)
attrs_mat = hcat(
    [cat_col(l, comms) for l in comm],
    [num_col(l) for l in comm])

attrs_mat = vcat(["nominal_feature" "numeric_feature"], attrs_mat)

writedlm(fn_out, attrs_mat, ',')
