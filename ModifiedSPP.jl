using Graphs, StatsBase, Random, BenchmarkTools, Distributions, Plots
include("Newman_Ziff.jl")

function sample_shortest_path(o::Int, d::Int, G::SimpleGraph, C::Int)
    n = nv(G)
    dist = fill(Inf, n)
    parents = [Set{}() for _ in 1:n]
    s = zeros(Int, n)

    dist[o] = 0
    s[o] = 1
    queue = [o]
    target_dist = Inf

    while !isempty(queue)
        current = popfirst!(queue)

        # Once we’re past the target level (dist[d]), break
        if dist[current] > C
            break
        end

        for neighbor in neighbors(G, current)
            if dist[neighbor] == Inf
                dist[neighbor] = dist[current] + 1
                push!(queue, neighbor)
                s[neighbor] += s[current]
                push!(parents[neighbor], current)
            elseif dist[neighbor] == dist[current] + 1
                s[neighbor] += s[current]
                push!(parents[neighbor], current)
            end
        end
    end

    if dist[d] == Inf || dist[d] > C
        return []
    end

    # Sample path from d to o
    path = [d]
    current = d
    while current != o
        extented_neighbour = Int64[]
        ps = parents[current]
        weight = [s[p] for p in ps]
        total = sum(weight)
        current = sample(ps, StatsBase.weights(weight ./ total))
        push!(path, current)
    end
    reverse!(path)
    return path
end

# Prepare the od set

function find_node_pairs_within_distance(g::Graph, C::Int)
    pairs = Set{Tuple{Int, Int}}()

    for u in 1:nv(g)
        visited = Dict{Int, Int}()  # node => distance
        queue = [(u, 0)]

        while !isempty(queue)
            (current, dist) = popfirst!(queue)
            if dist >= C
                continue
            end
            for v in neighbors(g, current)
                if !haskey(visited, v) || dist + 1 < visited[v]
                    visited[v] = dist + 1
                    push!(queue, (v, dist + 1))
                    if u < v  # avoid duplicates like (2,1) if (1,2) is already added
                        push!(pairs, (u, v))
                    end
                end
            end
        end
    end

    return collect(pairs)
end



# Geometric distribution
function sample_tau(L::Int, N::Int)
    p = 2L / (N * (N - 1))
    d = Geometric(p)
    return rand(d)  # returns τ ∈ {1, 2, 3, ...}
end


function run_single_spp(g::SimpleGraph, C::Int, OD_pairs::Vector{Tuple{Int,Int}})
    G = deepcopy(g)
    od_pairs = deepcopy(OD_pairs)
    N = nv(G)
    edge_list_tau = Vector{Tuple{Int,Int}}[]
    t = Int64[]
    # SSP model
    while od_pairs != []

        L = length(od_pairs)

        # sample the demand
        tau = sample_tau(L, N)

        # sample the origin and destination
        push!(t, tau)

        # sample the origin and destination
        x = rand(1:L)
        (o, d) = od_pairs[x]

        # sample the path
        path_old = sample_shortest_path(o, d, G, C)
        if isempty(path_old) 
            deleteat!(od_pairs, x)
            push!(edge_list_tau, [])
            continue
        end

        # remove edge from G 
        edge_list = Vector{Tuple{Int,Int}}()
        for i in 1:length(path_old)-1
            u = path_old[i]
            v = path_old[i+1]
            push!(edge_list, (u, v))
            rem_edge!(G, u, v)
        end
        push!(edge_list_tau, edge_list)

        # Check if od stays in the set
        if isempty(sample_shortest_path(o, d, G, C))
            # remove od from od set
            if x != length(od_pairs)
                od_pairs[x] = od_pairs[end]
            end    
            pop!(od_pairs)
        end
    end
    return t, edge_list_tau
end

function run_single_spp_newman(G::SimpleGraph, C, OD_pairs::Vector{Tuple{Int,Int}})
    _, edge_list_tau = run_single_spp(G, C, OD_pairs)
    edge_list = [(x...,) for ee in edge_list_tau for x in ee]
    edge_list = reverse(edge_list)
    windows = round.(Int,collect(range(1, ne(G)-1;length=50)))
    p_values, s_max_trial, chi_trial = run_single_trial(nv(G), edge_list, windows;shuffle=false)
    return p_values, s_max_trial, chi_trial
end


N = 10
C = 5
G_ER = erdos_renyi(N, Int(4 * N / 2))
OD_pairs = find_node_pairs_within_distance(G_ER, C)
@profview run_single_spp(G_ER, C, OD_pairs)

(o,d) = rand(OD_pairs)
sample_shortest_path(5, 9, G_ER, C)

using GraphPlot

gplot(G_ER, nodelabel=1:nv(G_ER))