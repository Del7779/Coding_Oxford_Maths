using Graphs, StatsBase, Random, BenchmarkTools, Distributions, Plots
include("Newman_Ziff.jl")

function sample_shortest_path_optimized(o::Int, d::Int, neighbors::Vector{Vector{Int}}, 
                                        removed_edges::BitSet, edge_dict::Dict{Tuple{Int,Int}, Int}, 
                                        C::Int, dist::Vector{Int}, parents::Vector{Vector{Int}}, 
                                        s::Vector{Int})
    fill!(dist, -1)
    empty!.(parents)
    fill!(s, 0)
    
    dist[o] = 0
    s[o] = 1
    queue = [o]
    q_idx = 1
    found = false

    while q_idx <= length(queue) && !found
        current = queue[q_idx]
        q_idx += 1

        current == d && (found = true; continue)
        dist[current] >= C && continue

        for neighbor in neighbors[current]
            e = (min(current, neighbor), max(current, neighbor))
            edge_num = edge_dict[e]
            edge_num in removed_edges && continue

            if dist[neighbor] == -1
                dist[neighbor] = dist[current] + 1
                push!(queue, neighbor)
                s[neighbor] = s[current]
                push!(parents[neighbor], current)
            elseif dist[neighbor] == dist[current] + 1
                s[neighbor] += s[current]
                push!(parents[neighbor], current)
            end
        end
    end

    found && dist[d] <= C || return Int[]
    
    # Backtrack using preallocated arrays
    path = Int[]
    current = d
    while current != o
        ps = parents[current]
        weights = s[ps]
        total = sum(weights)
        idx = sample(1:length(ps), Weights(weights, total))
        current = ps[idx]
        push!(path, current)
    end
    
    reverse!(path)
    push!(path, d)
    return path
end

function find_node_pairs_within_distance_optimized(g::SimpleGraph, C::Int)
    pairs = Tuple{Int,Int}[]
    Neighbors = [neighbors(g, u) for u in 1:nv(g)]
    visited = Vector{Int}(undef, nv(g))
    queue = Vector{Tuple{Int,Int}}(undef, nv(g)*C)
    
    for u in 1:nv(g)
        fill!(visited, -1)
        q_start = 1
        q_end = 1
        queue[1] = (u, 0)
        visited[u] = 0
        
        while q_start <= q_end
            (current, dist) = queue[q_start]
            q_start += 1
            
            dist < C || continue
            
            for v in Neighbors[current]
                if visited[v] == -1 || dist + 1 < visited[v]
                    visited[v] = dist + 1
                    q_end += 1
                    queue[q_end] = (v, dist + 1)
                    u < v && push!(pairs, (u, v))
                end
            end
        end
    end
    unique!(pairs)
    return pairs
end

function run_single_spp_optimized(g::SimpleGraph, C::Int, OD_pairs::Vector{Tuple{Int,Int}})
    n = nv(g)
    Neighbors = [neighbors(g, u) for u in 1:n]
    edge_list = edges(g) |> collect
    edge_dict = Dict((min(src(e), dst(e)), max(src(e), dst(e))) => i for (i, e) in enumerate(edge_list))
    removed_edges = BitSet()
    
    # Preallocate BFS buffers
    dist = Vector{Int}(undef, n)
    parents = [Int[] for _ in 1:n]
    s = Vector{Int}(undef, n)
    
    active_od = BitVector(ones(Bool, length(OD_pairs)))
    t = Int[]
    edge_list_tau = Vector{Tuple{Int,Int}}[]

    while any(active_od)
        L = sum(active_od)
        L == 0 && break
        
        # Sample tau using optimized geometric distribution
        τ = rand(Geometric(2L / (n * (n - 1)))) + 1
        push!(t, τ)
        
        # Select random active OD pair
        active_indices = findall(active_od)
        x = rand(active_indices)
        (o, d) = OD_pairs[x]
        
        # Find path with edge checking
        path = sample_shortest_path_optimized(o, d, Neighbors, removed_edges, edge_dict, C, dist, parents, s)
        
        if isempty(path)
            active_od[x] = false
            push!(edge_list_tau, [])
            continue
        end
        
        # Record removed edges
        edges_in_path = Tuple{Int,Int}[]
        for i in 1:length(path)-1
            u, v = path[i], path[i+1]
            e = (min(u, v), max(u, v))
            push!(edges_in_path, e)
            push!(removed_edges, edge_dict[e])
        end
        push!(edge_list_tau, edges_in_path)
        
        # Check connectivity using same BFS infrastructure
        fill!(dist, -1)
        dist[o] = 0
        queue = [o]
        q_idx = 1
        connected = false
        
        while q_idx <= length(queue) && !connected
            current = queue[q_idx]
            q_idx += 1
            
            current == d && (connected = true; break)
            dist[current] >= C && continue
            
            for neighbor in Neighbors[current]
                e = (min(current, neighbor), max(current, neighbor))
                edge_num = edge_dict[e]
                edge_num in removed_edges && continue
                
                if dist[neighbor] == -1
                    dist[neighbor] = dist[current] + 1
                    push!(queue, neighbor)
                end
            end
        end
        
        connected || (active_od[x] = false)
    end
    
    t, edge_list_tau
end

function run_single_spp_newman_optimized(G::SimpleGraph, C, OD_pairs::Vector{Tuple{Int,Int}})
    _, edge_list_tau = run_single_spp_optimized(G, C, OD_pairs)
    edge_list = reverse!([e for edges in edge_list_tau for e in reverse(edges)])
    windows = round.(Int, range(1, ne(G)-1, length=50))
    p_values, s_max_trial, chi_trial = run_single_trial(nv(G), edge_list, windows; shuffle=false)
    return p_values, s_max_trial, chi_trial
end


Ns = 10 .^[2,3,4]
C = 3
k = 4
times = Float64[]
for n in N
    m = Int(n*k/2)
    G = erdos_renyi(n, m)
    OD_pairs = find_node_pairs_within_distance_optimized(G, C)
    b = @benchmark run_single_spp_newman_optimized($G, $C, $OD_pairs)
    @info "Generating graph with N=$n, minimum time: $(minimum(b).time / 1e9)"
    push!(times, minimum(b).time / 1e9)
end

p = plot(Ns, times, 
    label="C=3",
    marker=:circle,
    xlabel="Network size (N)",
    ylabel="Time (seconds)",
    title="Time Complexity Analysis",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10)

# Fit and plot trend line
log_N = log.(Ns)
log_t = log.(times)
slope = (length(Ns) * sum(log_N .* log_t) - sum(log_N) * sum(log_t)) / 
        (length(Ns) * sum(log_N.^2) - sum(log_N)^2)

@info "Empirical complexity: O(N^$(round(slope, digits=2)))"
savefig(p, "Figure/SSP/time_complexity.png")