module Newman_Ziff
using Random, Statistics, StatsBase, Combinatorics, ProgressMeter, Distributed # Must be at top level

export run_single_trial

# ---------------------
# Union-Find with Size Tracking
# ---------------------
mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
end

function UnionFind(n::Int)
    return UnionFind(collect(1:n), ones(Int, n))
end

function uf_find(uf::UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]  # Path halving.
        x = uf.parent[x]
    end
    return x
end

function uf_find(uf::UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]  # Path halving.
        x = uf.parent[x]
    end
    return x
end

function uf_union!(uf::UnionFind, x::Int, y::Int)
    rx = uf_find(uf, x)
    ry = uf_find(uf, y)
    if rx == ry
        return
    end
    if uf.size[rx] < uf.size[ry]
        uf.parent[rx] = ry
        uf.size[ry] += uf.size[rx]
    else
        uf.parent[ry] = rx
        uf.size[rx] += uf.size[ry]
    end
end

# ---------------------
# Map an edge index to a unique pair (edge) of nodes.
# ---------------------
function index_to_edge_comb(index::Int, n::Int, m::Int)
    total_combinations = binomial(n, m)
    if index < 1 || index > total_combinations
        throw(ArgumentError("Index $index out of range (must be between 1 and $total_combinations)"))
    end
    c = Int[]
    j = -1
    index -= 1  # convert to 0-based index
    for s in 1:m
        cs = j + 1
        while index - binomial(n - 1 - cs, m - s) ≥ 0
            index -= binomial(n - 1 - cs, m - s)
            cs += 1
        end
        push!(c, cs)
        j = cs
    end
    return Tuple(c .+ 1)  # convert back to 1-based indexing
end

# ---------------------
# Sample a unique random edge index.
# ---------------------
function get_unique_edge!(used::Set{Int}, M::Int)
    idx = rand(1:M)
    while idx in used
        idx = rand(1:M)
    end
    push!(used, idx)
    return idx
end

# ---------------------
# Compute susceptibility.
# ---------------------
function compute_susceptibility(comp_sizes::Vector{Int})
    if isempty(comp_sizes)
        return 0.0
    end
    smax = maximum(comp_sizes)
    total = sum(comp_sizes) - smax
    if total == 0
        return 0.0
    end
    sumsq = sum(s^2 for s in comp_sizes if s != smax)
    return sumsq / total
end

# ---------------------
# Single-Trial Simulation for N=10^6
# ---------------------
"""
run_single_trial(N, window_fraction, max_points)

Runs one trial of the ER bond percolation simulation for N nodes.
A limited number of edges (N_limit) are added, and data is recorded in a window
around m = N/2. Returns a tuple: (p_values, s_max_trial, chi_trial)
"""

function run_single_trial(N::Int,edge_list::Vector,windows::Vector;shuffle::Bool=true)
    # Shuffle the edge list if specified
    if shuffle
        shuffle!(edge_list)
    end
    
    p_values = windows./length(edge_list)
    num_window_points = length(windows)
    uf = UnionFind(N)
    chi_trial = zeros(Float64, num_window_points)
    s_max_trial = zeros(Int, num_window_points)
    record_idx = 1
    for (num, (u,v)) in enumerate(edge_list)
        uf_union!(uf, u, v)
        if num in windows
            comp_sizes = Int[]
            largest_cluster = 1
            for i in 1:N
                if uf.parent[i] == i
                    size_val = uf.size[i]
                    push!(comp_sizes, size_val)
                    largest_cluster = max(largest_cluster, size_val)
                end
            end
            chi_trial[record_idx] = compute_susceptibility(comp_sizes)
            s_max_trial[record_idx] = largest_cluster
            record_idx += 1
        end
    end
    return p_values, s_max_trial, chi_trial
end


function Parallel_Newman_Ziff(N::Int, edge_list::Vector, windows::Vector; num_trials::Int=150)

# Wrap pmap with @showprogress to display progress.
trial_indices = 1:num_trials
results = @showprogress pmap(trial_indices) do trial
    run_single_trial(N, edge_list, windows)
end

# Collect results.
p_vals = results[1][1]  # assume p_values are the same for all trials
num_points = length(p_vals)
s_max_mat = zeros(Float64, num_trials, num_points)
chi_mat = zeros(Float64, num_trials, num_points)
for (i, res) in enumerate(results)
    s_max_mat[i, :] .= res[2]
    chi_mat[i, :] .= res[3]
end
s_max_avg = vec(mean(s_max_mat, dims=1))
chi_small = vec(mean(chi_mat, dims=1))
GCC_frac = s_max_avg ./ N

s_max_sq_mean = vec(mean(s_max_mat.^2, dims=1))
chi_giant = (s_max_sq_mean./ N^2 .- GCC_frac.^2) ./ GCC_frac

idx = argmax(chi_small)
p_c_est = p_vals[idx]
@info "Success with $(nprocs()) workers! For N=$N, estimated percolation threshold p_c ≈ $(round(p_c_est, digits=4))"
return p_vals, chi_giant, chi_small, GCC_frac, p_c_est

end

end
