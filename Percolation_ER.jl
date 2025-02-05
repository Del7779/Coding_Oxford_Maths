###############################################################################
# er_percolation.jl
#
# This script simulates bond percolation on an Erdős–Rényi (ER) graph using
# a Newman–Ziff–inspired method and performs data analysis and plotting.
#
# Required packages:
#   - Random
#   - Statistics
#   - Plots
#   - LsqFit
#
# To run:
#   julia er_percolation.jl
###############################################################################

using Random
using Statistics
using Plots
using Colors

plotlyjs() 
# using LsqFit

# -----------------------------
# Union-Find Data Structure
# -----------------------------
mutable struct UnionFind
    parent::Vector{Int}
    rank::Vector{Int}
end

# Constructor: initialize each node as its own parent; ranks start at zero.
function UnionFind(n::Int)
    parent = collect(1:n)    # Julia uses 1-indexing.
    rank = zeros(Int, n)
    return UnionFind(parent, rank)
end

# Find with path compression.
function uf_find(uf::UnionFind, x::Int)
    if uf.parent[x] != x
        uf.parent[x] = uf_find(uf, uf.parent[x])
    end
    return uf.parent[x]
end

# Union by rank.
function uf_union(uf::UnionFind, x::Int, y::Int)
    root_x = uf_find(uf, x)
    root_y = uf_find(uf, y)
    if root_x != root_y
        if uf.rank[root_x] > uf.rank[root_y]
            uf.parent[root_y] = root_x
        elseif uf.rank[root_x] < uf.rank[root_y]
            uf.parent[root_x] = root_y
        else
            uf.parent[root_y] = root_x
            uf.rank[root_x] += 1
        end
    end
end

# ------------------------------------------------------
# Newman-Ziff ER Percolation Simulation Function
# ------------------------------------------------------
"""
    newman_ziff_er_percolation_avg_degree(N; trials=1000, window_fraction=0.01)

Simulates bond percolation on an ER graph of `N` nodes over `trials` realizations.
Data is recorded in a window around `m = N/2` with relative half-width
`window_fraction`. Returns a tuple containing:
- p_values: the edge density values,
- var_smax: the standard deviation of the largest cluster size,
- chi_average: the averaged susceptibility of the reduced clusters,
- GCC_fraction: the fraction of nodes in the giant connected component.
"""
function newman_ziff_er_percolation_avg_degree(N::Int; trials::Int=1000, window_fraction::Float64=0.01)
    n = max(1, floor(Int, window_fraction * N))  # ensure at least 1
    num_window_points = 2 * n                      # window: from N/2 - n to N/2 + n

    # p_values spans the edge density in the window.
    p_values = range((N/2 - n) / N, stop = (N/2 + n) / N, length = num_window_points)

    # Allocate arrays to store the largest cluster sizes and the susceptibility values.
    s_max_values = zeros(Float64, trials, num_window_points)
    chi_values   = zeros(Float64, trials, num_window_points)

    # Loop over trials.
    for t in 1:trials
        # Create a list of all possible edges in the complete graph.
        edges = [(i, j) for i in 1:N for j in (i+1):N]
        shuffle!(edges)

        uf = UnionFind(N)
        record_idx = 1  # index in the recording window
        edge_count = 0  # current edge count

        for (u, v) in edges
            edge_count += 1
            uf_union(uf, u, v)

            # Only record data in the chosen window.
            if (edge_count > (N/2 - n)) && (edge_count <= (N/2 + n))
                # Compute cluster sizes.
                component_sizes = zeros(Int, N)
                for i in 1:N
                    root = uf_find(uf, i)
                    component_sizes[root] += 1
                end
                largest_cluster = maximum(component_sizes)
                # Get the sizes of all clusters except the largest.
                reduced_cluster = component_sizes[component_sizes .!= largest_cluster]

                # Compute susceptibility: weighted second moment divided by the sum of sizes.
                chi = 0.0
                if sum(reduced_cluster) != 0
                    unique_vals = unique(reduced_cluster)
                    counts_arr = [count(==(val), reduced_cluster) for val in unique_vals]
                    chi = sum(counts_arr .* (unique_vals .^ 2)) / sum(reduced_cluster)
                end

                chi_values[t, record_idx] = chi
                s_max_values[t, record_idx] = largest_cluster
                record_idx += 1

                if record_idx > num_window_points
                    break
                end
            elseif edge_count > (N/2 + n)
                break
            end
        end
    end

    # Compute averages over trials.
    chi_average   = vec(mean(chi_values, dims = 1))
    s_max_average = vec(mean(s_max_values, dims = 1))
    GCC_fraction  = s_max_average / N

    # Compute the standard deviation of s_max.
    s_max_sq_mean = vec(mean(s_max_values .^ 2, dims = 1))
    var_smax = sqrt.(s_max_sq_mean .- s_max_average .^ 2)

    return collect(p_values), var_smax, chi_average, GCC_fraction
end

# -----------------------------
# Example: Single Run (N=1000)
# -----------------------------
system_size = 1000
trials = 100
p_vals, susceptibilities_1, susceptibilities_2, GCC_fraction = newman_ziff_er_percolation_avg_degree(system_size; trials = trials, window_fraction = 0.1)

# Create two separate plots.
p_single_1 = plot(p_vals, GCC_fraction,
    marker = :circle, linestyle = :solid,
    label = "GCC Fraction",
    title = "ER Percolation for N=$system_size",
    xlabel = "p (edge density)", ylabel = "GCC Fraction")

p_single_2 = plot(p_vals, susceptibilities_2,
    marker = :circle, linestyle = :solid,
    label = "Susceptibility",
    title = "ER Percolation for N=$system_size",
    xlabel = "p (edge density)", ylabel = "Susceptibility")

display(p_single_1)
display(p_single_2)

# -------------------------------------------------
# Example: Simulation for Different System Sizes
# -------------------------------------------------
system_sizes = [1000, 3000, 5000, 8000, 12000, 18000]
critical_points = Float64[]
simulation_data = Dict{Int, Dict{String, Any}}()

@showprogress for N in system_sizes
    p_vals, susceptibilities_1, susceptibilities_2, GCC_fraction = newman_ziff_er_percolation_avg_degree(N; trials = 100, window_fraction = 0.1)
    simulation_data[N] = Dict(
        "p_vals"            => p_vals,
        "susceptibilities_1" => susceptibilities_1,
        "susceptibilities_2" => susceptibilities_2,
        "GCC_fraction"      => GCC_fraction
    )
    # Heuristic: compute the discrete derivative d(GCC_fraction)/dp.
    dGCC_dp = diff(GCC_fraction) ./ diff(p_vals)
    # Find the index where the derivative is maximal.
    idx = findmax(dGCC_dp)[2]
    p_c = p_vals[idx]
    push!(critical_points, p_c)
    println("Estimated percolation threshold (p_c) for ER graph with N=$N: $(round(p_c, digits = 4))")
end

# Plot critical points vs. 1/N.
p_critical = plot(1.0 ./ system_sizes, critical_points,
    marker = :circle, linestyle = :solid,
    label = "Critical Points",
    xlabel = "1/N", ylabel = "p_c",
    title = "Critical Points vs 1/N")
display(p_critical)

# Create three plots for the simulation data.
p_sim1 = plot(title = "Variance of the reduced cluster size", xlabel = "p", ylabel = "Susceptibilities 1")
p_sim2 = plot(title = "GCC Fraction scaled", xlabel = "p", ylabel = "GCC Fraction * N^(1/3)")
p_sim3 = plot(title = "Variance of the GCC", xlabel = "p", ylabel = "Susceptibilities 2")

for N in system_sizes
    data = simulation_data[N]
    # Color based on system size (red channel varies between 0 and 1).
    color_val = RGB(N / maximum(system_sizes), 0.0, 0.0)
    plot!(p_sim1, data["p_vals"], data["susceptibilities_1"],
        seriestype = :scatter, mc = color_val, label = "N=$N",ms=2)
    plot!(p_sim2, data["p_vals"], data["GCC_fraction"] .*N^(1/3),
        linestyle = :solid, color = color_val, label = "N=$N",lw = 1)
    plot!(p_sim3, data["p_vals"], data["susceptibilities_2"],
        seriestype = :scatter, mc = color_val, label = "N=$N",ms=2)
end

display(p_sim1)
display(p_sim2)
display(p_sim3)

