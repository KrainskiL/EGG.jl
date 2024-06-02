######################
# Auxilary functions #
######################
"""
    dist(i::Int, j::Int, embed::Array{Float64,2})

Calculates Euclidian distance between two vectors from embedding array.

**Arguments**
* `v1::Int` index of first vector
* `v2::Int` index of second vector
* `embed::Array{Float64,2}` graph embedding array
"""
function dist(v1::Int, v2::Int, embed::Array{Float64,2})
    if v1 == v2
        return 0.0
    else
        return @inbounds sqrt(sum(i -> (embed[v1, i] - embed[v2, i])^2, axes(embed, 2)))
    end
end

"""
    JS(vC::Vector{Float64}, vB::Vector{Float64},
        vI::Vector{Int}, internal::Int, vLen::Int)

Jensen-Shannon divergence with Dirichlet-like prior.

**Arguments**
* `vC::Vector{Float64}` first distribution of edges within and between communities
* `vB::Vector{Float64}` second distribution of edges within and between communities
* `vI::Vector{Int}` indicator of internal (1) and external (0) edges w.r.t. communities, if empty compute overall JS distance
* `internal::Int` internal JS distance switch, if 1 return internal, else return external
"""
function JS(vC::Vector{Float64}, vB::Vector{Float64},
    vI::AbstractVector{Bool}, internal::Bool)

    if !isempty(vI)
        inter = vI .== internal
        vect_p = vC[inter]
        vect_q = vB[inter]
    else
        vect_p = copy(vC)
        vect_q = copy(vB)
    end
    sp_1 = sum(vect_p) + length(vect_p)
    sp_2 = sum(vect_q) + length(vect_q)
    @. vect_p = (vect_p + 1) / sp_1
    @. vect_q = (vect_q + 1) / sp_2
    vect_m = (vect_p .+ vect_q) ./ 2
    f = sum(@. vect_p * log(vect_p / vect_m) + vect_q * log(vect_q / vect_m))
    return f / 2
end

"""
Change 2-dimensional into 1-dim index
"""
function idx(n::Int, i::Int, j::Int)
    return n * (i - 1) - (i - 1) * (i - 2) ÷ 2 + j - i + 1
end

parse_flag(flag::String) = !isnothing(findfirst(==(flag), ARGS)) ? true : false

function parseargs()
    try
        ###Flags###

        # Check if calculations should be verbose
        verbose = parse_flag("-v")
        # Check if provided graph is directed
        directed = parse_flag("-d")

        ##Output file name base
        idx = findfirst(==("-o"), ARGS)
        @assert !isnothing(idx) "Output filename is required"
        fn_out = ARGS[idx+1]

        ###Edgelist###

        idx = findfirst(==("-g"), ARGS)
        @assert !isnothing(idx) "Edgelist file is required"
        fn_edges = ARGS[idx+1]
        @assert isfile(fn_edges) "$fn_edges is not a file"
        println("The graph is $(fn_edges)")

        # Read edges
        edges = readdlm(fn_edges, Float64)
        rows, no_cols = size(edges)
        verbose && println("$no_cols columns and $rows rows in edgelist file.")

        # Validate file structure
        @assert no_cols == 2 || no_cols == 3 "Expected 2 or 3 columns in edgelist file"
        v_min = minimum(edges[:, 1:2])
        @assert v_min == 0 || v_min == 1 "Vertices should be either 0-based or 1-based"

        # Make vertices 1-based
        if v_min == 0.0
            edges[:, 1:2] .+= 1.0
        end
        no_vertices = Int(maximum(edges[:, 1:2]))
        verbose && println("Graph contains $no_vertices vertices")

        # If graph is unweighted, add unit weights
        # Compute vertices weights
        vweight = zeros(no_vertices)
        eweights = no_cols == 2 ? ones(rows) : edges[:, 3]
        edges = convert.(Int, edges[:, 1:2])
        for i in 1:rows
            vweight[edges[i, 1]] += eweights[i]
            vweight[edges[i, 2]] += eweights[i]
        end
        verbose && println("Done preparing edgelist and vertices weights")

        ###Communities###

        idx = findfirst(==("-c"), ARGS)
        @assert !isnothing(idx) "Communities list is required"
        fn_comm = ARGS[idx+1]
        comm = readdlm(fn_comm, Int)
        @assert isfile(fn_comm) "$fn_comm is not a file"
        comm_rows, no_cols = size(comm)

        # Validate file structure
        @assert comm_rows == no_vertices "No. communities ($comm_rows) differ from no. nodes ($no_vertices)"
        @assert no_cols == 1 || no_cols == 2 "Expected 1 or 2 columns in communities file, but encountered $no_cols."
        # 2 columns file - sort by first column and extract only second column
        if no_cols == 2
            comm = comm[sortperm(comm[:, 1]), 2]
            comm = reshape(comm, size(comm)[1], 1)
        end
        c_min = minimum(comm)
        @assert c_min == 0 || c_min == 1 "Communities should be either 0-based or 1-based, but are $c_min based."

        # Make communities 1-based
        if c_min == 0
            comm .+= 1
        end
        verbose && println("Done preparing communities.")

        ###Embedding###

        idx = findfirst(==("-e"), ARGS)
        @assert !isnothing(idx) "Embedding file is required"
        fn_embed = ARGS[idx+1]
        @assert isfile(fn_embed) "$fn_embed is not a file"

        # Read embedding
        embedding = []
        try
            embedding = readdlm(fn_embed, Float64)
        catch
            verbose && println("Embedding in node2vec format. Loading without first line.")
            embedding = readdlm(fn_embed, Float64, skipstart=1)
        end
        # Validate file
        @assert no_vertices == size(embedding, 1) "No. rows in embedding and no. vertices in a graph differ."

        # If embedding contains indices in first column, sort by it and remove the column
        try
            order = convert.(Int, embedding[:, 1])
            verbose && println("Sorting embedding by first column")
            embedding = embedding[sortperm(order), 2:end]
        catch

        end
        verbose && println("Done preparing embedding.")

        ###Node attributes###

        idx = findfirst(==("-a"), ARGS)
        if !isnothing(idx)
            fn_attr = ARGS[idx+1]
            @assert isfile(fn_attr) "$fn_attr is not a file"
            attrs_df = CSV.read(fn_attr, DataFrame)
            @assert no_vertices == nrow(attrs_df) "No. rows in attributes DataFrame and no. vertices in a graph differ."
        else
            attrs_df = nothing
        end

        ##Seed
        idx = findfirst(==("--seed"), ARGS)
        seed = !isnothing(idx) ? parse(Int, ARGS[idx+1]) : -1

        ##Attributes target sample size
        idx = findfirst(==("--attr-target-size"), ARGS)
        attr_sample_size = !isnothing(idx) ? parse(Int, ARGS[idx+1]) : 5

        ##Target transfer ratio
        idx = findfirst(==("--attr-target-ratio"), ARGS)
        attr_transfer_ratio = !isnothing(idx) ? parse(Float64, ARGS[idx+1]) : 0.5
        @assert 0.0 <= attr_transfer_ratio <= 1.0 "Attribute transfer ratio must be between 0 and 1"

        ##No. graphs
        idx = findfirst(==("-n"), ARGS)
        n = !isnothing(idx) ? parse(Int, ARGS[idx+1]) : 10

        return edges, eweights, vweight, comm, embedding, attrs_df, fn_out, verbose, directed, seed, attr_sample_size, attr_transfer_ratio, n
    catch e
        showerror(stderr, e)
        println("\n\nUsage:")
        println("\tjulia CGE_CLI.jl -g edgelist -e embedding -c communities -a node_attributes -o outfile [--seed seed] [-v] [-d]")
        println("\nParameters:")
        println("edgelist: rows should contain two whitespace separated vertices ids (edge) and optional weights in third column")
        println("embedding: rows should contain whitespace separated embeddings of vertices")
        println("communities: rows should contain cluster identifiers of vertices with optional vertices ids in the first column")
        println("node_attributes: CSV file of node attributes, numeric and categorical features are supported")
        println("outfile: basename for output files, .edgelist extension will be aded for list of generated edges and .attr will be added for generated attributes")
        println("seed: RNG seed for local measure sampling")
        println("-v: flag for debugging messages")
        println("-d: flag for usage of directed framework")
        exit(1)
    end
end
