###############################################################################
# er_percolation.jl
#
# This script simulates bond percolation on an Erdős–Rényi (ER) graph using
# a Newman–Ziff–inspired method and performs data analysis and plotting.
using Random
using Statistics
using Plots
using Colors
using ProgressMeter
using BenchmarkTools
using StatsBase  
using Pkg
using Combinatorics

combinations(1, 2)
collect(combinations(1, 20))


using Combinatorics  # Required for binomial coefficient binomial(n, m)

function index_to_edge_comb(index::Int, n::Int, m::Int)
    """
    Generate a hyperedge from an index given the number of nodes and hyperedge size.

    Parameters:
    -----------
    index :: Int  → The 1-based index of the hyperedge in lexicographic order.
    n :: Int  → The number of nodes.
    m :: Int  → The hyperedge size.

    Returns:
    --------
    Vector{Int} → The hyperedge corresponding to the given index.

    Example:
    --------
    index_to_edge_comb(3, 4, 3)  # Returns [0, 2, 3]
    """
    total_combinations = binomial(n, m)

    if index < 1 || index > total_combinations
        throw(ArgumentError("Index $index out of range (must be between 1 and $total_combinations)"))
    end

    c = Int[] # Tuple to store the selected hyperedge
    j = -1  # Tracks the last selected node
    index -= 1  # Convert 1-based index to 0-based for combinatorial calculations

    for s in 1:m
        cs = j + 1
        while index - binomial(n - 1 - cs, m - s) ≥ 0
            index -= binomial(n - 1 - cs, m - s)
            cs += 1
        end
        push!(c, cs)
        j = cs
    end

    return Tuple(c .+ 1) # Convert 0-based to 1-based indexing
end

index_to_edge_comb(2, 4, 2)  # Returns [0, 1, 2]

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

# This newman shuffle is basically a Fisher-Yates shuffle （Shuffle）
function neman_shuffle(N::Int)
    edges = [(i,j) for i in 1:N for j in (i+1):N]
    M = length(edges)
    for i in 1:M
        j = rand(i:M)
        edges[i], edges[j] = edges[j], edges[i]
    end
    return edges
end

# -----------------------------
# A dynamical way to shuffle
# Compute the greatest common divisor.
function gcd(a::Int, b::Int)
    b == 0 ? a : gcd(b, a % b)
end

# Pick a random integer in 1:(T-1) that is coprime with T.
function random_coprime(T::Int)
    while true
        a = rand(1:T-1)
        if gcd(a, T) == 1
            return a
        end
    end
end

random_coprime(6)
# Given an index r (0-indexed), convert it to the corresponding edge (i, j)
# in a complete graph with N nodes. The edges are assumed to be ordered lexicographically.
function edge_unrank(r::Int, N::Int)
    # We wish to find i such that:
    #   S(i) = (i-1)*N - ((i-1)*i) ÷ 2 ≤ r < S(i+1)
    lo = 1
    hi = N + 1  # hi is one past the last valid i
    while lo < hi
        mid = (lo + hi) ÷ 2
        S_mid = (mid - 1) * N - div((mid - 1) * mid, 2)
        if S_mid <= r
            lo = mid + 1
        else
            hi = mid
        end
    end
    i = lo - 1
    S_i = (i - 1) * N - div((i - 1) * i, 2)
    j = i + 1 + (r - S_i)
    return (i, j)
end

# Generator that produces all edges in a random order using a linear congruential permutation.
const used_ab_pairs = Set{Tuple{Int, Int}}()

function shuffled_edges(N::Int)
    T = div(N * (N - 1), 2) 
    a = 0::Int
    b = 0::Int # Total number of edges in a complete graph on N nodes
    while true
        a = random_coprime(T)  # Choose a random multiplier that is coprime with T
        b = rand(0:T-1)        # Choose an arbitrary offset
        if (a, b) ∉ used_ab_pairs
            push!(used_ab_pairs, (a, b))  # Store the used pair
            break
        end
    end
    # println("a: $a, b: $b")
    # The mapping r -> mod(a*r + b, T) defines a permutation of 0:(T-1).
    return ( edge_unrank(mod(a * r + b, T), N) for r in 0:(T-1) )
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

function susceptibility(sizes::Vector{Int})
    if isempty(sizes)
        return 0.0
    end
    smax = maximum(sizes)
    reduced = filter(s -> s != smax, sizes)
    if isempty(reduced)
        return 0.0
    end
    # Define susceptibility as sum(s^2) / sum(s) for the reduced clusters.
    return sum(x -> x^2, reduced) / sum(reduced)
end

function newman_ziff_er_percolation_avg_degree(N::Int; trials::Int=1000, window_fraction::Float64=0.01, max_points::Int=1000)
    # Compute window parameters.
    n = max(1, floor(Int, window_fraction * N))
    full_points = 2 * n
    num_window_points = min(full_points, max_points)
    sampling_interval = full_points / num_window_points  # e.g. if full_points=2000 and max_points=1000, sampling_interval=2

    # Define the range of p-values (edge densities) for the recording window.
    p_values = range((N / 2 - n) / N, stop=(N / 2 + n) / N, length=num_window_points)

    # Allocate arrays to store the largest cluster sizes and susceptibility values.
    s_max_values = zeros(Float64, trials, num_window_points)
    chi_values = zeros(Float64, trials, num_window_points)

    # Loop over trials.
    for t in 1:trials
        uf = UnionFind(N)  
        
        # Use the on‐the‐fly shuffled generator of edges.
        # edges = shuffled_edges(N)
        # edges = neman_shuffle(N)
         # Your union–find data structure (assumed to be defined elsewhere
        record_idx = 1      # index in the sampling window (1 .. num_window_points)
        sample_counter = 0  # counts how many edges in the window have been processed
        index_track = Set{Int}()
        # Process edges sequentially.
        N_limit::Int = N/2 + n
        M::Int = N * (N - 1) / 2
        for edge_count in 1:N_limit
            index_random = rand(1:M)
            while index_random in index_track  # Ensure uniqueness
                index_random = rand(1:M)
            end
            push!(index_track, index_random)
            (u,v) = index_to_edge_comb(index_random, N, 2)
            uf_union(uf, u, v)
            # Only record data if edge_count is within the recording window.
            if (edge_count > (N / 2 - n)) && (edge_count <= (N / 2 + n))
                sample_counter += 1
                # Check if it's time to record a sample.
                if sample_counter ≥ sampling_interval * record_idx
                    # Compute cluster sizes.
                    component_sizes = zeros(Int, N)
                    for i in 1:N
                        root = uf_find(uf, i)
                        component_sizes[root] += 1
                    end
                    component_sizes = filter(x -> x != 0, component_sizes)
                    largest_cluster = maximum(component_sizes)
                    chi = susceptibility(component_sizes)

                    chi_values[t, record_idx] = chi
                    s_max_values[t, record_idx] = largest_cluster
                    record_idx += 1
                    # Stop if we've collected all the desired samples.
                    if record_idx > num_window_points
                        break
                    end
                end
            end
        end
    end

    # Compute averages over trials.
    chi_average = vec(mean(chi_values, dims=1))
    s_max_average = vec(mean(s_max_values, dims=1))
    GCC_fraction = s_max_average / N

    # Compute the standard deviation of s_max.
    s_max_sq_mean = vec(mean(s_max_values .^ 2, dims=1))
    var_smax = sqrt.(s_max_sq_mean .- s_max_average .^ 2)

    return collect(p_values), var_smax, chi_average, GCC_fraction
end

# -----------------------------
# Example: Single Run (N=1000)
# start_time = time_ns()
# system_size = 10000
# trials = 100
# p_vals, susceptibilities_1, susceptibilities_2, GCC_fraction = newman_ziff_er_percolation_avg_degree(system_size; trials=trials, window_fraction=0.4)
# # Record the end time in nanoseconds
# end_time = time_ns()
# # Calculate the elapsed time in seconds
# elapsed_time = (end_time - start_time) / 1e9
# # Create two separate plots.
# p_single_1 = plot(p_vals, GCC_fraction,
#     marker=:circle, linestyle=:solid,
#     label="GCC Fraction",
#     title="ER Percolation for N=$system_size",
#     xlabel="p (edge density)", ylabel="GCC Fraction")

# p_single_2 = plot(p_vals, susceptibilities_1,
#     marker=:circle, linestyle=:solid,
#     label="Susceptibility",
#     title="ER Percolation for N=$system_size",
#     xlabel="p (edge density)", ylabel="Susceptibility")

# display(p_single_1)
# display(p_single_2)

# -------------------------------------------------
# Example: Simulation for Different System Sizes
# -------------------------------------------------
system_sizes = [100, 1000, 10000, 20000,100000,800000]
system_sizes = [5000]

# 14hours for N= 200000
critical_points = Float64[]
simulation_data = Dict{Int,Dict{String,Any}}()

@showprogress for N in system_sizes
    # Notice the btime has its own scope.
    start_time = time_ns()
    p_vals, susceptibilities_1, susceptibilities_2, GCC_fraction = newman_ziff_er_percolation_avg_degree(N; trials=1000, window_fraction=0.4)
    simulation_data[N] = Dict(
        "p_vals" => p_vals,
        "susceptibilities_1" => susceptibilities_1,
        "susceptibilities_2" => susceptibilities_2,
        "GCC_fraction" => GCC_fraction
    )
    # Find the index where the derivative is maximal.
    idx = argmax(susceptibilities_2)
    p_c = p_vals[idx]
    push!(critical_points, p_c)
    println("Estimated percolation threshold (p_c) for ER graph with N=$N: $(round(p_c, digits = 4))")
    # Record the end time in nanoseconds
    end_time = time_ns()
    # Calculate the elapsed time in seconds
    elapsed_time = (end_time - start_time) / 1e9
    println("Elapsed time: $elapsed_time seconds")
end

























# -----------------------------
# Plotting
# -----------------------------
# Plot critical points vs. 1/N.
p_critical = plot(1.0 ./ system_sizes, critical_points,
    marker=:circle, linestyle=:solid,
    label="Critical Points",
    xlabel="1/N", ylabel="p_c",
    title="Critical Points vs 1/N")
display(p_critical)
 
using Plots
using PyPlot
pyplot()
# Uncomment the following line to switch to the PyPlot backend:
# pyplot()

# Create four empty plots with titles, axis labels, and high resolution.
p_sim1 = plot(title="Variance of the reduced cluster size", legend=:outerright, xlabel="p", ylabel="Susceptibilities 1", dpi=300)
p_sim2 = plot(title="GCC Fraction scaled", xlabel="p", ylabel="GCC Fraction * N^(1/3)", dpi=300, legend=:outerright)
p_sim3 = plot(title="Variance of the GCC", xlabel="p", ylabel="Susceptibilities 2", dpi=300, legend=:outerright)
p_sim4 = plot(title="S vs (p-pc)", xlabel="p-pc", ylabel="GCC Fraction", dpi=300, legend=:outerright)

for N in system_sizes
    data = simulation_data[N]
    # Choose a color that varies with system size (e.g., varying the red channel).
    color_val = RGB(N / maximum(system_sizes), 0.0, 0.0)
    
    # Plot on p_sim1: empty markers with outlines and a connecting line.
    plot!(p_sim1, data["p_vals"], data["susceptibilities_1"],
        label = "N=$N",
        linecolor = color_val, lw = 1,
        marker = (:circle, 3, :white, stroke(0.2, color_val)))
    
    # Plot on p_sim2: same marker style.
    plot!(p_sim2, data["p_vals"], data["GCC_fraction"]*N^(1/3),
        label = "N=$N",
        linecolor = color_val, lw = 1,
        marker = (:circle, 3, :white, stroke(0.2, color_val)))
    
    # Plot on p_sim3: same marker style.
    plot!(p_sim3, data["p_vals"], data["susceptibilities_2"],
        label = "N=$N",
        linecolor = color_val, lw = 1,
        marker = (:circle, 3, :white, stroke(0.2, color_val)))
    
    # Here we want to plot on p_sim4 using only the data for which (p_vals - 1/2) > 1/2.
    # First, compute (p_vals - 1/2) and get indices where the condition holds.
    idx = findall(x -> x > 1/2, data["p_vals"])
    
    # Extract the filtered x and y data.
    filtered_p_vals = data["p_vals"][idx]
    filtered_GCC = data["GCC_fraction"][idx]
    
    # Only plot if there is data to plot.
    if !isempty(filtered_p_vals)
        plot!(p_sim4, filtered_p_vals, filtered_GCC,
            label = "N=$N",
            linecolor = color_val, lw = 1,
            marker = (:circle, 3, :white, stroke(0.2, color_val)),
            xaxis = :log10, yaxis = :log10)
    else
        @warn "No points passed the filter for system size N=$N."
    end
end

xx = 10 .^ (-0.2:0.01:-0.1)
plot(p_sim4,xx, xx.*10^(-1),lw=2, label="slope = 1")
display(p_sim1)
display(p_sim2)
display(p_sim3)
display(p_sim4)





# Save the plots to disk.
savefig(p_critical, "Figure/critical_points_vs_N.png")
savefig(p_sim1, "Figure/variance_reduced_cluster_size.png")
savefig(p_sim2, "Figure/GCC_fraction_scaled.png")
savefig(p_sim3, "Figure/variance_GCC.png")


