###############################
# Weighted Geometric Chung Lu #
###############################
"""
wGCL(edges::Array{Int,2}, eweights::Vector{Float64}, comm::Array{Int,2},
            embed::Array{Float64,2}, distances::Vector{Float64}, outfile::String, verbose::Bool = false)

Calculates Weighted Geometric Chung-Lu model and divergence score for graph and embedding.

**Arguments**
* `edges::Array{Int,2}` array with edges definition
* `eweights::Vector{Float64}` edges weights
* `vweights::Vector{Float64}` nodes weights
* `comm::Array{Int,2}` assignment of vertices to communities
* `embed::Array{Float64,2}` array with vertices embeddings
* `attrs_df::Union{DataFrame,Nothing}` DataFrame with node attributes
* `base_outfile::String` base pattern for output files
* `n_samples::Int` number of generated graphs
* `seed::Int` RNG seed for local measure score
* `verbose::Bool` verbose switch, if true prints additional processing information
"""
function wGCL(edges::Array{Int,2},
    eweights::Vector{Float64},
    vweights::Vector{Float64},
    comm::Matrix{Int},
    embed::Matrix{Float64},
    attrs_df::Union{DataFrame,Nothing},
    base_outfile::String,
    n_samples::Int,
    attr_sample_size::Int=5,
    attr_transfer_ratio::Float64=0.5,
    seed::Int=-1,
    verbose::Bool=false)

    # Default parameters values
    start = time()
    epsilon = 0.25 # learning rate in Chung Lu model
    delta = 0.001 # desired precision of degree estimation in Chung Lu model
    AlphaMax = 10.0 # upper bound of alpha search
    AlphaStep = 0.25 # step in alpha search
    alpha_div_counter = 5 # early stopping threshold (iterations of alpha search without improvement)
    skip_div = false # skipping flags based on early stopping counters
    no_vertices = maximum(edges)
    no_edges = size(edges, 1)
    verbose && println("Graph has $no_vertices vertices and $no_edges edges")

    # Read communities
    @assert size(comm, 1) == no_vertices "No. rows in communities list not matching no. vertices"
    n_parts = maximum(comm)
    verbose && println("Graph has $n_parts communities")

    # Extend embedding if attributes were given
    if !isnothing(attrs_df)
        emb_attr = deepcopy(attrs_df)
        cat_cols = filter(col -> eltype(emb_attr[!, col]) <: AbstractString, names(emb_attr))
        #One-hot encoding
        for col in cat_cols
            for cat in unique(emb_attr[!, col])
                emb_attr[!, col*"_"*cat] = Float64.(emb_attr[!, col] .== cat)
            end
        end
        embed = hcat(embed, Matrix(select!(emb_attr, Not(cat_cols))))
        #Standardizing
        embed = StatsBase.transform(fit(ZScoreTransform, embed, dims=1), embed)
    end
    # Compute C-vector
    vect_len = Int(n_parts * (n_parts + 1) / 2)
    vect_C = zeros(Float64, vect_len)
    vect_B = zeros(Float64, vect_len)

    for i in 1:no_edges
        a = eweights[i]
        j, k = extrema([comm[edges[i, 1]], comm[edges[i, 2]]])
        vect_C[idx(n_parts, j, k)] += a
    end

    # Intra-community indicator
    vect_I = falses(vect_len)
    j = 1
    for i in 1:n_parts
        vect_I[j] = true
        j += (n_parts - i + 1)
    end
    best_div = typemax(Float64)
    best_alpha = -1.0
    dim = size(embed, 2)
    verbose && println("Embedding has $dim dimensions")

    # Compute Euclidean distance vector D[] given the embedding
    p_len = no_vertices * (no_vertices + 1) ÷ 2
    D = zeros(Float64, p_len)
    for i in 1:no_vertices
        for j in i:no_vertices
            l = idx(no_vertices, i, j)
            if i == j
                D[l] = 0.0
            else
                D[l] = dist(i, j, embed)
            end
        end
    end
    lo, hi = extrema(D)
    D = (D .- lo) ./ (hi - lo) # normalize to [0,1]

    # Chung Lu model weights to be tuned
    T = ones(no_vertices)

    for alpha in AlphaStep:AlphaStep:(AlphaMax+delta)
        write(stderr, ".")
        # Apply kernel (g(dist))
        GD = zeros(Float64, p_len)
        for i in 1:no_vertices
            for j in i:no_vertices
                k = idx(no_vertices, i, j)
                GD[k] = (1 - D[k])^alpha # transform w.r.t. alpha
            end
        end
        # Learn GCL model numerically
        diff = 1.0
        while diff > delta # stopping criterion
            S = zeros(no_vertices)
            for i in 1:no_vertices
                for j in i:no_vertices
                    tmp = T[i] * T[j] * GD[idx(no_vertices, i, j)]
                    S[i] += tmp
                    if i != j
                        S[j] += tmp
                    end
                end
            end
            f = 0.0
            for i in 1:no_vertices
                move = epsilon * T[i] * (vweights[i] / S[i] - 1.0)
                T[i] += move
                f = max(f, abs(vweights[i] - S[i])) # convergence w.r.t. degrees
            end
            diff = f
            # verbose && println("diff = $diff")
        end
        # Compute probas P[]
        P = zeros(Float64, p_len)
        for i in 1:no_vertices
            for j in i:no_vertices
                k = idx(no_vertices, i, j)
                P[k] = T[i] * T[j] * GD[k]
            end
        end

        if !skip_div
            # Compute B-vector given P[] and comm[]
            vect_B = zeros(vect_len)
            for i in 1:no_vertices
                for j in i:no_vertices
                    k, l = extrema([comm[i], comm[j]])
                    vect_B[idx(n_parts, k, l)] += P[idx(no_vertices, i, j)]
                end
            end
            f = JS(vect_C, vect_B, Bool[], true)

            if f < best_div
                best_div = f
                best_alpha = alpha
                alpha_div_counter = 5
            else
                alpha_div_counter -= 1
                skip_div = alpha_div_counter == 0
            end
        end
        skip_div && break
    end
    write(stderr, "\n")
    GD = zeros(Float64, p_len)
    for i in 1:no_vertices
        for j in i:no_vertices
            k = idx(no_vertices, i, j)
            GD[k] = (1 - D[k])^best_alpha # transform w.r.t. alpha
        end
    end
    # Learn GCL model numerically
    diff = 1.0
    while diff > delta # stopping criterion
        S = zeros(no_vertices)
        for i in 1:no_vertices
            for j in i:no_vertices
                tmp = T[i] * T[j] * GD[idx(no_vertices, i, j)]
                S[i] += tmp
                if i != j
                    S[j] += tmp
                end
            end
        end
        f = 0.0
        for i in 1:no_vertices
            move = epsilon * T[i] * (vweights[i] / S[i] - 1.0)
            T[i] += move
            f = max(f, abs(vweights[i] - S[i])) # convergence w.r.t. degrees
        end
        diff = f
        # println("diff = $diff")
    end
    # Compute probas P[]
    P = zeros(Float64, p_len)
    for i in 1:no_vertices
        for j in i:no_vertices
            k = idx(no_vertices, i, j)
            P[k] = T[i] * T[j] * GD[k]
        end
    end
    for i in 1:no_vertices
        P[idx(no_vertices, i, i)] = 0.0
    end
    exp_edges = sum(P)
    println("Expected no edges is $exp_edges while actual no edges is $no_edges")
    P = P .* no_edges / exp_edges
    print("Model training took: $(time()-start)")
    times = []
    function agg_attr(col)
        if eltype(col) <: AbstractString
            return mode(col)
        else
            return mean(col)
        end
    end
    function combine_node_attrs(row_source::DataFrameRow, row_target::DataFrameRow, transfer_ratio::Float64)
        transfer_ratio == 0 && return row_source
        transfer_ratio == 1.0 && return row_target
        row_names = names(row_source)
        row_values = []
        for col in row_names
            if typeof(row_source[col]) <: Number
                push!(row_values, transfer_ratio * row_target[col] + (1 - transfer_ratio) * row_source[col])
            else
                push!(row_values, sample([row_target[col], row_source[col]], Weights([transfer_ratio, 1 - transfer_ratio])))
            end
        end
        return NamedTuple{Tuple(Symbol.(row_names))}(row_values)
    end
    for iter in 1:n_samples
        start = time()
        Random.seed!(iter)
        gengraph = []
        for i in 1:no_vertices
            for j in i:no_vertices
                if i == j
                    continue
                end
                k = idx(no_vertices, i, j)
                if rand() < P[k]
                    push!(gengraph, (i, j))
                end
            end
        end
        push!(times, time() - start)
        no_gen_edges = length(gengraph)
        writedlm(base_outfile * "_" * string(iter) * ".edgelist", gengraph, ' ')
        # Attributes generation
        if !isnothing(attrs_df)
            new_attr_df = DataFrame()
            attrs_df[!, "_comm"] = vec(comm)
            gdf = groupby(attrs_df, "_comm")
            for row in eachrow(attrs_df)
                comm_attrs = gdf[(row."_comm",)]
                n = nrow(comm_attrs)
                sampled_attrs = comm_attrs[rand(1:n, min(n, attr_sample_size)), :]
                target_attrs = combine(sampled_attrs, All() .=> agg_attr, renamecols=false)[1,:]
                new_row = combine_node_attrs(row, target_attrs, attr_transfer_ratio)
                push!(new_attr_df, new_row)
            end
            CSV.write(base_outfile * "_" * string(iter) * ".attrs", select(new_attr_df, Not("_comm")))
        end
        println("Finished $(iter) main pass with $(no_gen_edges) edges")
    end
    println("Graph generation took on average: $(mean(times)) with std: $(std(times))")
    return nothing
end

"""
wGCL_directed(edges::Array{Int,2}, weights::Vector{Float64}, comm::Array{Int,2},
            embed::Array{Float64,2}, distances::Vector{Float64}, outfile::String, verbose::Bool = false)

Calculates directed Weighted Geometric Chung-Lu model and divergence score for graph and embedding.

**Arguments**
* `edges::Array{Int,2}` array with edges definition (two whitespace separated vertices ids)
* `eweights::Vector{Float64}` edges weights
* `comm::Array{Int,2}` assignment of vertices to communities
* `embed::Array{Float64,2}` array with vertices embeddings
* `distances::Vector{Float64}` distances between vertices
* `vweights::Vector{Float64}` landmarks total weights - used only with landmarks approximation
* `init_vweights::Vector{Float64}` vector with original (full) vertices weights - used only with landmarks approximation
* `v_to_l::Vector{Int}` mapping from vertices to landmarks (landmarks membership) - used only with landmarks approximation
* `init_edges::Array{Int,2}` array with original (full) graph edges - used only with landmarks approximation
* `init_eweights::Vector{Float64}` vector with original (full) edges weights - used only with landmarks approximation
* `init_embed::Matrix{Float64}` array with original embedding for full graph - used only with landmarks approximation
* `split::Bool` indicator for splitting JS divergence score (global score)
* `seed::Int` RNG seed for local measure score
* `auc_samples::Int` no. samples for local measure score
* `verbose::Bool` verbose switch, if true prints additional processing information
"""
function wGCL_directed(edges::Array{Int,2}, eweights::Vector{Float64}, comm::Matrix{Int},
    embed::Matrix{Float64}, distances::Vector{Float64}, vweights::Vector{Float64},
    init_vweights::Vector{Float64}, v_to_l::Vector{Int}, init_edges::Array{Int,2},
    init_eweights::Vector{Float64}, init_embed::Matrix{Float64},
    split::Bool, seed::Int=-1, auc_samples::Int=10000, verbose::Bool=false)
    # Default values
    delta = 0.001 # desired precision of degree estimation in Chung Lu model
    AlphaMax = 10.0 # upper bound of alpha search
    AlphaStep = 0.25 # step in alpha search
    alpha_div_counter = alpha_auc_counter = 5 # early stopping threshold (iterations of alpha search without improvement)
    skip_div = skip_auc = false # skippig flags based on early stopping counters

    no_vertices = maximum(edges)
    no_edges = size(edges, 1)
    verbose && println("auc_samples: $auc_samples")
    landmarks = !isempty(v_to_l)

    verbose && println("Graph has $no_vertices vertices and $no_edges edges")
    landmarks && verbose && println("Original graph has $(maximum(init_edges)) vertices and $(size(init_edges,1)) edges")

    # Read communities
    @assert size(comm, 1) == no_vertices "No. communities not matching no. vertices"
    n_parts = maximum(comm)
    verbose && println("Graph has $n_parts communities")

    # Compute degrees
    degree_in = zeros(no_vertices)
    degree_out = zeros(no_vertices)
    star_check = zeros(Int64, no_vertices)
    for i in 1:no_edges
        a = eweights[i]
        v1 = edges[i, 1]
        v2 = edges[i, 2]
        degree_out[v1] += a
        degree_in[v2] += a
        star_check[v1] += 1
        star_check[v2] += 1
    end

    # Check for edge case - star graph
    is_star = false
    # Star based on either in or out edges
    if !isnothing(findfirst(==(no_vertices - 1), star_check)) && sum(star_check) == 2 * (no_vertices - 1)
        is_star = true
        # Star based on both in and out edges
    elseif !isnothing(findfirst(==(2 * (no_vertices - 1)), star_check)) && sum(star_check .== 2) == no_vertices - 1
        is_star = true
    end
    verbose && is_star && println("Graph is a star in respect to either in or out edges")

    if is_star
        return [-1.0, zeros(5)...]
    end

    # Compute C-vector
    vect_len = Int(n_parts * n_parts)
    vect_C = zeros(Float64, vect_len)
    vect_B = zeros(Float64, vect_len)

    for i in 1:no_edges
        j = comm[edges[i, 1]]
        k = comm[edges[i, 2]]
        vect_C[(j-1)*n_parts+k] += eweights[i]
    end

    # Indicator - internal to a community
    vect_I = falses(vect_len)
    for i in 1:(n_parts+1):vect_len
        vect_I[i] = true
    end
    best_div = best_div_ext = best_div_int = best_auc_err = best_auc = typemax(Float64)
    best_alpha = best_alpha_auc = -1.0
    dim = size(embed, 2)
    verbose && println("Embedding has $dim dimensions")
    # Loop over alpha's

    # Compute Euclidean distance vector D[] given embed and alpha
    p_len = no_vertices * (no_vertices + 1) ÷ 2
    D = zeros(Float64, p_len)

    #Read distances
    @assert size(distances, 1) == no_vertices "Distances vector length is not equal to no. vertices"
    for i in 1:no_vertices
        for j in i:no_vertices
            l = idx(no_vertices, i, j)
            if i == j
                D[l] = distances[i]
            else
                D[l] = dist(i, j, embed)
            end
        end
    end
    lo, hi = extrema(D)
    D = (D .- lo) ./ (hi - lo) # normalize to [0,1]

    adj_no_vertices = no_vertices
    adj_edges = edges
    adj_eweights = eweights
    if landmarks
        adj_no_vertices = length(init_vweights)
        adj_edges = init_edges
        adj_eweights = init_eweights
    end

    if landmarks
        adj_p_len = adj_no_vertices * (adj_no_vertices + 1) ÷ 2
        full_graph_D = zeros(Float64, adj_p_len)
        for i in 1:adj_no_vertices
            for j in (i+1):adj_no_vertices
                l = idx(adj_no_vertices, i, j)
                full_graph_D[l] = dist(i, j, init_embed)
            end
        end
        lo, hi = extrema(full_graph_D)
        full_graph_D = (full_graph_D .- lo) ./ (hi - lo) # normalize to [0,1]
    end
    # Loop here - exclude Alpha = 0
    Tin = ones(no_vertices)
    Tin[degree_in.==0] .= 0.0
    Tout = ones(no_vertices)
    Tout[degree_out.==0] .= 0.0

    # Generate arrays of edges and non-edges
    NE = Tuple{Int64,Int64}[]
    for i in 1:adj_no_vertices
        for j in 1:adj_no_vertices
            if i != j
                push!(NE, (i, j))
            end
        end
    end

    ## tuples of edges
    E = Tuple{Int64,Int64,Float64}[]
    for (i, e) in enumerate(eachrow(adj_edges))
        push!(E, tuple(e..., adj_eweights[i]))
    end

    ## tuples of non-edges
    NE = collect(setdiff(Set(NE), Set([e[1:2] for e in E])))

    for alpha in AlphaStep:AlphaStep:(AlphaMax+delta)
        write(stderr, ".")
        # Apply kernel (g(dist))
        GD = zeros(Float64, p_len)
        for i in 1:no_vertices
            for j in i:no_vertices
                k = idx(no_vertices, i, j)
                GD[k] = (1 - D[k])^alpha # transform w.r.t. alpha
            end
        end
        # Learn GCL model numerically
        diff = 1.0
        epsilon = 0.9
        while diff > delta # stopping criterion
            Sin = zeros(no_vertices)
            Sout = zeros(no_vertices)
            for i in 1:no_vertices
                for j in i:no_vertices
                    k = idx(no_vertices, i, j)
                    tmp1 = Tin[i] * Tout[j] * GD[k]
                    tmp2 = Tin[j] * Tout[i] * GD[k]
                    Sin[i] += tmp1
                    Sin[j] += tmp2
                    Sout[i] += tmp2
                    Sout[j] += tmp1
                end
            end

            f = 0.0
            for i in 1:no_vertices
                if degree_in[i] > 0
                    Tin[i] += epsilon * Tin[i] * (degree_in[i] / Sin[i] - 1.0)
                    f = max(f, abs(degree_in[i] - Sin[i])) # convergence w.r.t. degrees
                end
                if degree_out[i] > 0
                    Tout[i] += epsilon * Tout[i] * (degree_out[i] / Sout[i] - 1.0)
                    f = max(f, abs(degree_out[i] - Sout[i])) # convergence w.r.t. degrees
                end
            end
            if f > diff
                epsilon *= 0.99
            end
            diff = f
            # verbose && println("diff= $diff")
        end

        # Compute probas P[]
        P = zeros(Float64, 2 * p_len)
        for i in 1:no_vertices
            for j in 1:no_vertices
                a, b = extrema([i, j])
                P[no_vertices*(i-1)+j] = Tout[i] * Tin[j] * GD[idx(no_vertices, a, b)]
            end
        end

        if !skip_auc
            pos = zeros(auc_samples)
            neg = zeros(auc_samples)
            auc_weights = zeros(auc_samples)
            if landmarks
                ## random positive cases
                seed != -1 && Random.seed!(seed)
                for (ind, edge) in enumerate(sample(E, auc_samples, replace=true))
                    i, j, w = edge
                    a, b = extrema((i, j))
                    adj_Tout = Tout[v_to_l[i]] * init_vweights[i] / vweights[v_to_l[i]]
                    adj_Tin = Tin[v_to_l[j]] * init_vweights[j] / vweights[v_to_l[j]]
                    pos[ind] = adj_Tout * adj_Tin * ((1 - full_graph_D[idx(adj_no_vertices, a, b)])^alpha)
                    auc_weights[ind] = w
                end
                ## random negative cases
                seed != -1 && Random.seed!(seed)
                for (ind, edge) in enumerate(sample(NE, auc_samples, replace=true))
                    i, j = edge
                    a, b = extrema(edge)
                    adj_Tout = Tout[v_to_l[i]] * init_vweights[i] / vweights[v_to_l[i]]
                    adj_Tin = Tin[v_to_l[j]] * init_vweights[j] / vweights[v_to_l[j]]
                    neg[ind] = adj_Tout * adj_Tin * ((1 - full_graph_D[idx(adj_no_vertices, a, b)])^alpha)
                end
            else
                ## random positive cases
                seed != -1 && Random.seed!(seed)
                for (ind, edge) in enumerate(sample(E, auc_samples, replace=true))
                    i, j, w = edge
                    pos[ind] = P[no_vertices*(i-1)+j]
                    auc_weights[ind] = w
                end
                pos = [P[no_vertices*(i-1)+j] for (i, j) in sample(E, auc_samples, replace=true)]
                ## random negative cases
                seed != -1 && Random.seed!(seed)
                neg = [P[no_vertices*(i-1)+j] for (i, j) in sample(NE, auc_samples, replace=true)]
            end

            # 1 - AUC
            auc = 1 - sum((pos .> neg) .* auc_weights) / sum(auc_weights)

            if auc < best_auc
                best_auc = auc
                best_auc_err = 1.96 * sqrt(auc * (1 - auc) / auc_samples) ## error from 95% CI
                best_alpha_auc = alpha
                alpha_auc_counter = 5
            else
                alpha_auc_counter -= 1
                skip_auc = alpha_auc_counter == 0
            end
        end

        if !skip_div
            # Compute B-vector given P[] and comm[]
            vect_B = zeros(vect_len)
            for i in 1:no_vertices
                for j in 1:no_vertices
                    m = (comm[i] - 1) * n_parts + comm[j]
                    vect_B[m] += P[no_vertices*(i-1)+j]
                end
            end
            if !split
                f = JS(vect_C, vect_B, Bool[], true)
            else
                x = JS(vect_C, vect_B, vect_I, true)
                y = JS(vect_C, vect_B, vect_I, false)
                f = (x + y) / 2.0
            end
            if f < best_div
                best_div = f
                best_alpha = alpha
                best_div_ext = !split ? 0.0 : x
                best_div_int = !split ? 0.0 : y
                alpha_div_counter = 5
            else
                alpha_div_counter -= 1
                skip_div = alpha_div_counter == 0
            end
        end
        skip_div && skip_auc && break
    end
    write(stderr, "\n")
    return [best_alpha, best_div, best_div_ext, best_div_int, best_alpha_auc, best_auc, best_auc_err]
end
