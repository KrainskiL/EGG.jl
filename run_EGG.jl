using EGG

edges, eweights, vweight, comm, embedding, attrs_df, fn_out, verbose, directed, seed, attr_sample_size, attr_transfer_ratio, n = parseargs()

if directed
    results = wGCL_directed(edges, eweights, comm, embedding, attrs_df, fn_out, n, attr_sample_size, attr_transfer_ratio, seed, verbose)
else
    results = wGCL(edges, eweights, vweight, comm, embedding, attrs_df, fn_out, n, attr_sample_size, attr_transfer_ratio, seed, verbose)
end