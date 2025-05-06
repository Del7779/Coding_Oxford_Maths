using Random, Graphs, LsqFit, StatsBase, Combinatorics, Plots
using Markdown, LaTeXStrings
using Measures
using Random
using StatsBase
using Printf
using Plots
using SparseArrays
using Arpack
using LinearAlgebra

#
include("Newman_Ziff.jl")

# Message passing algorithm for bond percolation
@doc raw"""
A message passing algorithm for calculating the fraction of giant component and the average cluster size of a graph in bond percolation.
It is essentially a discreat dynamical systems of 4m dimensions with m being the number of edges.
Given:
  - `adj`: adjacency list for an undirected graph
  - `p`: bond (edge) occupation probability
  - `z`: formal variable (z=1 for giant component analysis)
  - `tol`: convergence tolerance
  - `maxiter`: maximum number of MP iterations
Returns a dictionary containing the second derivative of ``\\H_{i<-j}\\``.
"""
function discreate_ds(adj::Vector, p::Float64; tol=1e-8, maxiter=3000, z = 1.0)

    # Dictionary to store messages H_{i<-j}
    N = length(adj)
    messages = Dict{Tuple{Int,Int}, Float64}()
    messages_prime = Dict{Tuple{Int,Int}, Float64}()

    # Initialize all messages to 1.0
    for i in 1:N
        for j in adj[i]
            Random.seed!(42)
            messages_prime[(i, j)] = rand()
            messages[(i, j)] = rand()
        end
    end

    # Iterative Hij and Hij_prime simultanepusly, although this can be done separately since hij is not dependent on hij_prime
    delta_history = Dict{Tuple{Int,Int},Float64}()
    delta_prime_history = Dict{Tuple{Int,Int},Float64}()
    for it in 1:maxiter
        delta = 0.0
        for i in 1:N 
            for j in adj[i]
                oldval = messages[(i, j)]
                oldval2 = messages_prime[(i, j)]
                # Compute product over neighbors of j except i
                prodval = 1.0
                prodval_1 = 0.0
                for k in adj[j]
                    if k != i
                        prodval *= messages[(j, k)]
                        prodval_1 += messages_prime[(j, k)]/messages[(j, k)]
                    end
                end
                # Update rule   
                newval = (1 - p) + z * p * prodval
                newval2 = p * (1 + prodval_1) * prodval   
                messages[(i, j)] = newval
                messages_prime[(i, j)] = newval2
                diff1 = abs(newval - oldval)
                diff2 = abs(newval2 - oldval2)
                delta_history[(i,j)] = diff1
                delta_prime_history[(i, j)] = diff2 
                diff = maximum([diff1,diff2])
                if diff > delta
                    delta = diff
                end
            end
        end
                
        # for i in 1:N
        #     for j in adj[i]
        #         oldval= messages_prime[(i, j)]
        #         # Compute product over neighbors of j except i
        #         prodval_1 = 0.0
        #         prodval_2 = 1.0
        #         for k in adj[j]
        #             if k != i
        #                 prodval_1 += messages_prime[(j, k)]/messages[(j, k)]
        #                 prodval_2 *= messages[(j, k)]
        #             end
        #         end
        #         # Update rule
        #         newval = p * (1 + prodval_1) * prodval_2
        #         messages_prime[(i, j)] = newval

        #         # Track maximum change for convergence
        #         diff = abs(newval - oldval)
        #         if diff > delta
        #             delta = diff
        #         end
        #     end
        # end

        # Stop if converged
        if delta < tol
            println("For P = $p, both converged after $it iterations")
            return messages, messages_prime, delta_history, delta_prime_history
            break
        elseif it == maxiter
            println("For P = $p, both did not converge after $maxiter iterations")
            return messages, messages_prime, delta_history, delta_prime_history
        end
    end
end

function Average_ns(hij,hij_prime,adj)
    N = length(adj)

    # calculate the S 
    non_perc_prob = Float64[]
    for i in 1:N
        val = 1.0
        for j in adj[i]
            val *= hij[(i,j)]
        end
        push!(non_perc_prob,val)
    end
    mean_non_perc_prob = mean(non_perc_prob)

    # calculate the average_n
    val = 0.0
    for i in 1:N
        val1 = 0.0
        
        for j in adj[i]
            val2 = 1.0

            for k in adj[i]
                if k != j
                    val2 *= hij[(i,k)]
                end
            end

            val1 += hij_prime[(i,j)] * val2
        end
        val += val1
    end
    average_n = 1 + 1/(N*mean_non_perc_prob) * val

    return average_n
end


# Function to calculate the size of the giant component
function giant_component_size(adj, messages)
    N = length(adj)
    total = 0.0
    for i in 1:N
        # Probability i is NOT in the giant component:
        not_in_gc = 1.0
        for j in adj[i]
            not_in_gc *= messages[(i, j)]
        end
        total += (1.0 - not_in_gc)  # Probability i IS in giant comp
    end
    return total / N
end

# Run MP for a range of p values
function MP_GCC_ns(adj, p)
    # messages = Hij(adj, p)
    messages, messages_prime, _ , _ = discreate_ds(adj, p)  
    return giant_component_size(adj, messages), Average_ns(messages,messages_prime, adj)
end

# Some basic functions to work with graphs
function Edge_list(G::SimpleGraph)
    edge_list = Vector{Tuple{Int,Int}}()
    for e in edges(G)
        push!(edge_list, (e.src, e.dst))
    end
    return edge_list
end

function adjacencies(G::SimpleGraph)
    adj = Vector{Vector{Int}}(undef, nv(G))
    for i in 1:nv(G)
        adj[i] = (neighbors(G, i))
    end
    return adj
end

## Collect graph data
function graph_data(G::SimpleGraph)
    edge_list = Edge_list(G)
    adj = adjacencies(G)
    deg = degree(G)
    return edge_list, adj, deg
end

# Some useful functions to generate graphs
## Generate a Bethe lattice (tree) with given coordination number and depth
function bethe_tree(z::Int, depth::Int)
    # z is the coordination number (degree of non-leaf nodes)
    # depth is the number of levels in the tree
    
    # Calculate total number of nodes
    N = 1
    for d in 1:(depth-1)
        N += (z-1)^(d)
    end
    
    G = SimpleGraph(N)
    
    # Start with root node (1)
    current_idx = 1
    next_idx = 2
    
    # For each level
    for level in 1:(depth-1)
        # Get number of nodes at this level
        nodes_at_level = (z-1)^(level-1)
        
        # For each node at current level
        for node in current_idx:(current_idx + nodes_at_level - 1)
            # Add z-1 children
            for child in 1:(z-1)
                if next_idx <= N
                    add_edge!(G, node, next_idx)
                    next_idx += 1
                end
            end
        end
        current_idx += nodes_at_level
    end
    
    return G
end

"""
    powerlaw_exp_distribution(τ, κ, kmax) -> p::Vector{Float64}

Returns a probability vector p[1..kmax], where p[k] is proportional to
k^(-τ) * exp(-k/κ), normalized so that sum(p) == 1.
"""
function powerlaw_exp_distribution(τ::Float64, κ::Float64, kmax::Int)
    # Unnormalized probabilities
    raw = [k^(-τ) * exp(-k/κ) for k in 1:kmax]
    # Normalize
    Z = sum(raw)
    return raw ./ Z
end

"""
    sample_degree_sequence(N, τ, κ, kmax) -> deg_seq::Vector{Int}

Samples a degree sequence of length N from the distribution
p_k ∝ k^(-τ)*exp(-k/κ) (k=1..kmax).
"""
function sample_degree_sequence(N::Int, τ::Float64, κ::Float64, kmax::Int)
    p = powerlaw_exp_distribution(τ, κ, kmax)  # p[k] for k=1..kmax
    cdf = cumsum(p)                             # cdf[k] = sum of p[1..k]

    deg_seq = zeros(Int, N)
    while sum(deg_seq) % 2 != 0 || sum(deg_seq) == 0 # Ensure even sum
        deg_seq = Vector{Int}(undef, N)
        for i in 1:N
            r = rand()          # uniform random in [0,1)
            k = searchsortedfirst(cdf, r)
            deg_seq[i] = k
        end
    end

    return deg_seq
end


## Test code 

# Create a Bethe lattice with z=3 (coordination number) and depth=5
g_tree = bethe_tree(3, 20)


# ER Random graph
N = 10000
k = 5
M = Int(N*k/2)
# Generate a random graph with N nodes and M edges
g_ER = erdos_renyi(N, M)
# get rid of isolated nodes
# isolated_nodes = findall(degree(ER) .== 0)
# for node in isolated_nodes
#     rem_vertex!(ER, node)
# end


# BA Scale-free graph
g_BA = barabasi_albert(10000, 5, 4)

# Configuration model
k_max = 1000
N = 100000
degree_sequence = sample_degree_sequence(N, 2.5, 10.0, k_max)
CONFM = random_configuration_model(length(degree_sequence),degree_sequence)
edge_list, adj, deg = graph_data(CONFM);

## Real world data set 
function load_undirected_graph(filename::String)
    # Use a Set to store unique undirected edges
    edges = Set{Tuple{Int,Int}}()
    
    open(filename, "r") do io
        for line in eachline(io)
            # Skip comment lines
            if startswith(line, "#")
                continue
            end
            parts = split(line)
            if length(parts) < 2
                continue
            end
            u = parse(Int, parts[1])
            v = parse(Int, parts[2])
            # Store edge with lower ID first to treat (u,v) and (v,u) as the same
            edge = u <= v ? (u, v) : (v, u)
            push!(edges, edge)
        end
    end

    # Determine all unique nodes (assumes node IDs are positive integers)
    nodes = Set{Int}()
    for (u, v) in edges
        push!(nodes, u)
        push!(nodes, v)
    end
    num_nodes = maximum(nodes)  # assuming nodes are numbered from 1 up to max
    
    # Create an undirected graph with num_nodes vertices
    g = SimpleGraph(num_nodes)
    for (u, v) in edges
        add_edge!(g, u, v)
    end
    
    return g, edges
end

# Load the graph from the file
filename = "p2p-Gnutella04.txt"
g_p2p, unique_edges = load_undirected_graph(filename)







# estimate the critical point using non-backtracking
function non_backtracking_critical_point(g_p2p::SimpleGraph)
    N = nv(g_p2p)
    adj_m = adjacency_matrix(g_p2p)
    d = degree(g_p2p)
    # Diagonal degree matrix
    D = spdiagm(d)

    # Identity
    # Create identity matrix of size N x N
    I_n = sparse(I, N, N)

    # Alternative M 
    NB_m = [adj_m (I_n .- D); I_n spzeros(N,N)]

    vals, vecs = eigs(NB_m, nev=1, which=:LM)

    Pc = 1/real(vals[1])
    return Pc
end









# MP method
## Input a graph g
# function gcc_chi_plot(g::SimpleGraph)
g = g_tree
edge_list, adj, deg = graph_data(g)
N = nv(g)
k = mean(degree(g))
non_backtracking_critical_point(g_ER)

windows = round.(Int,collect(range(1, length(edge_list)-1;length=50)))
p = windows ./ length(edge_list)
# p = collect(0.5:0.001:0.9)
s = zeros(Float64, (2,length(p)))
for (i, p_t) in enumerate(p)
    s[1,i],s[2,i] = MP_GCC_ns(adj, p_t)
end

# Numerical simulation
p_vals, chi_giant, chi_small, GCC_frac, p_c_est = Parallel_Newman_Ziff(N, edge_list, windows; num_trials=100);

# plot
h = plot(title="Power Law Network", xlabel="p", ylabel="Giant Component Size", size=(800,600), dpi=600, margin=5mm, 
    tickfontsize=12, guidefontsize=14, legendfontsize=12);
h2 = twinx()
scatter!(h, p_vals, chi_small, ylabel="χ", markersize=6, markerstrokecolor=:blue, label="Simulation N = $N, <k> = $(round(k;digits=2))", markercolor=RGBA(1,1,1,0), color=:blue, marker=:circle);
scatter!(h2, p_vals, GCC_frac, ylabel="Fraction of GCC", label="", markersize=6, color=:white, markerstrokecolor=:red, marker=:square,ylims=(0,1))
# vline!(h2,[non_backtracking_critical_point(g)], label="Estimated Critical Point", linestyle=:dash, color=:black, linewidth=2)
plot!(h,p,s[2,:], label="", color=:blue, linewidth=1);
plot!(h2,p,s[1,:], label="", color=:red, linewidth=1, right_margin=5mm)
display(h)

# save
# savefig(h,"Figure/Message_passing/Gcc_Chi.png")

# end

gcc_chi_plot(g_tree)












# A naive simulation just to compare against
# Parameters
ps = p    # occupation probabilities
num_realizations = 200     # number of trials per p

# Generate the full ER graph once
g_full = erdos_renyi(N, M)

# Function to compute susceptibility χ
function compute_susceptibility(g::SimpleGraph)
    comps = connected_components(g)
    sizes = [length(c) for c in comps]
    max_size = maximum(sizes)
    finite_sizes = filter(x -> x < max_size, sizes)
    if isempty(finite_sizes)
        return 0.0
    end
    chi = sum(x^2 for x in finite_sizes) / sum(finite_sizes)
    return chi
end

# Run simulations and average
chis_mean = Float64[]

for p in ps
    chis_p = Float64[]  # store chi for this p across trials

    for trial in 1:num_realizations
        g_perc = SimpleGraph(N)

        for e in edges(g_full)
            if rand() < p
                add_edge!(g_perc, src(e), dst(e))
            end
        end

        chi = compute_susceptibility(g_perc)
        push!(chis_p, chi)
    end

    avg_chi = mean(chis_p)
    push!(chis_mean, avg_chi)
    @printf("p = %.4f, ⟨χ⟩ = %.2f\n", p, avg_chi)
end

# Plot the averaged result
h1 = plot() 
scatter!(h1,ps, chis_mean,
    xlabel = "Occupation Probability p",
    ylabel = "Average Susceptibility ⟨χ⟩",
    title = "Bond Percolation on ER Graph (N = $N, ⟨k⟩ = $k_avg",
    lw = 2, label = "Simulation(Naive)", legend = :topright)
plot!(h1,p,s[2,:], label="Message Passing", color=:blue, linewidth=1,ylims=(0,6))
scatter!(h1,p_vals, chi_small, label="Simulation (Newman)", color=:red, linewidth=1)
