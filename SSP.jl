using Distributed
using Plots
using ProgressMeter
using BenchmarkTools
using Graphs
using Random, Distributions
using StatsBase
include("Newman_Ziff.jl")
        
function sample_shortest_path(o::Int, d::Int, G::SimpleGraph, C::Int)
    n = nv(G)
    dist = fill(Inf, n)
    parents = [[] for _ in 1:n]
    s = zeros(Int, n)

    dist[o] = 0
    s[o] = 1
    queue = [o]

    target_dist = Inf

    while !isempty(queue)
        current = popfirst!(queue)

        # Once we’re past the target level (dist[d]), break
        if dist[current] > target_dist || dist[current] > C
            break
        end

        for neighbor in neighbors(G, current)
            if dist[neighbor] == Inf
                dist[neighbor] = dist[current] + 1
                push!(queue, neighbor)
                s[neighbor] += s[current]
                push!(parents[neighbor], current)
                if neighbor == d
                    target_dist = dist[neighbor]  # mark when d is first found
                end
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
        ps = parents[current]
        weight = [s[p] for p in ps]
        total = sum(weight)
        current = sample(ps,StatsBase.weights(weight ./ total))
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


N = 2048
# C = 3
G = erdos_renyi(N, Int(4 * N / 2))

using Profile
using ProfileCanvas

# Profile a single run
od_pairs = find_node_pairs_within_distance(G, 3)
Profile.clear()
@profile run_single_spp(G, 3, od_pairs)

# Save profile results to HTML
ProfileCanvas.html() # Opens in default browser

C = [1,2,3,10, 20, 40, 80, 1500]
times = Float64[]
for c in C
    od_pairs = find_node_pairs_within_distance(G, c)
    
    b = @benchmark run_single_spp_newman($G, $c, $od_pairs)

    t = minimum(b).time / 1e9  # Convert to seconds
    
    push!(times, t)
end

p = plot(C, times, 
    label="N=1000",
    marker=:circle,
    xlabel="Cost threshold (C)",
    ylabel="Time (seconds)",
    title="Time Complexity Analysis",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,dpi=600)
# Fit and plot trend line
log_C = log.(C)
log_t = log.(times)
slope = (length(C) * sum(log_C .* log_t) - sum(log_C) * sum(log_t)) / 
        (length(C) * sum(log_C.^2) - sum(log_C)^2)

@info "Empirical complexity: O(N^$(round(slope, digits=2)))"
savefig(p, "time_complexityvsC.png")


Ns = 10 .^[2,3,4]
C = 3
times = Float64[]
for n in Ns
    m = Int(n*4/2)
    G = erdos_renyi(n, m)
    OD_pairs = find_node_pairs_within_distance(G, C)
    b = @benchmark run_single_spp($G, $C, $OD_pairs)
    @info "Generating graph with N=$n, minimum time: $(minimum(b).time / 1e9)"
    push!(times, minimum(b).time / 1e9)
end

log_N = log.(Ns)
log_t = log.(times)
slope = (length(Ns) * sum(log_N .* log_t) - sum(log_N) * sum(log_t)) / 
        (length(Ns) * sum(log_N.^2) - sum(log_N)^2)


# Plot the results
p = plot(Ns, times, 
    label="C=3, Slope = $(round(slope, digits=2))",
    marker=:circle,
    xlabel="Network size (N)",
    ylabel="Time (seconds)",
    title="Time Complexity Analysis",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10)

savefig(p,"Figure/SSP/time_complexity.png")











N = 1000
C = 3
G_ER = erdos_renyi(N, Int(4 * N / 2))
num_simulations = 100
windows = round.(Int,collect(range(1, ne(G_ER)-1;length=50)))
p_values = windows./(ne(G_ER))
S_loop = zeros(Float64, num_simulations, length(windows));
χ_loop = zeros(Float64, num_simulations, length(windows));

h1 = plot(title="Susceptibility ",
    xlabel="p", ylabel="χ", legend=:topleft, grid=true, size=(800, 600),    
    xguidefontsize=14, 
    yguidefontsize=14,
    legendfontsize=12,
    xtickfontsize=12,
    ytickfontsize=12,dpi=600);

h2 = plot(title=" Size of the Giant Component",
    xlabel="p", ylabel="S", legend=:topleft, grid=true, size=(800, 600),
    xguidefontsize=14, 
    yguidefontsize=14,
    legendfontsize=12,
    xtickfontsize=12,
    ytickfontsize=12,dpi=600);



@info "Running simulation for C = $c"
OD_pairs = find_node_pairs_within_distance(G_ER, C)
@showprogress for i in 1:num_simulations
t, edge_list_tau = run_single_spp(G_ER, c,OD_pairs)
edge_list = [(x...,) for ee in edge_list_tau for x in ee]
reverse!(edge_list)
_, s_max_trial, chi_trial = run_single_trial(nv(G_ER), edge_list, windows,shuffle=false)     
S_loop[i, :] = s_max_trial
χ_loop[i, :] = chi_trial
end
S = vec(mean(S_loop, dims=1)) / nv(G_ER)
χ = vec(mean(χ_loop, dims=1))
plot!(h2,1 .- p_values, S, label="(C=$C) (no weights)", xlabel="p", ylabel="S")
plot!(h1,1 .- p_values, χ, label="(C=$C) (no weights)")



display(h1)
display(h2)