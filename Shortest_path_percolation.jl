using Distributed
using Plots
using ProgressMeter
using BenchmarkTools
include("Newman_Ziff.jl")

# get the number of available CPU threads
# nmax_cpu = Sys.CPU_THREADS
# if nprocs() ==1
#     addprocs(nmax_cpu)
# end


@everywhere begin
    using Graphs
    using Random, Distributions
    using StatsBase
    include("Newman_Ziff.jl")  
         
    function sample_shortest_path(o::Int, d::Int, G::SimpleGraph, C::Int)
        n = nv(G)
        dist = fill(Inf, n)
        parents = [[] for _ in 1:n]
        s = zeros(Int, n)
    
        queue = [o]
        dist[o] = 0
        s[o] = 1
    
        while !isempty(queue)

            current = popfirst!(queue)

            if current == d
                continue
             end

            if dist[current] >= C
                continue
            end
            for neighbor in neighbors(G, current)
                if dist[neighbor] == Inf
                    # First time visiting this node
                    dist[neighbor] = dist[current] + 1
                    push!(queue, neighbor)
                    s[neighbor] += s[current]
                    push!(parents[neighbor], current)
                elseif dist[neighbor] == dist[current] + 1
                    # Another shortest path to neighbor
                    s[neighbor] += s[current]
                    push!(parents[neighbor], current)
                end
            end
        end
    
        if dist[d] == Inf || dist[d] > C
            return []
        end
    
        # Now do a weighted backwalk from d to o to sample a path
        path = [d]
        current = d
        while current != o
            ps = parents[current]
            weight = [s[p] for p in ps]
            total = sum(weight)
            probs = StatsBase.weights(weight ./ total)
            idx = sample(1:length(ps), probs)
            current = ps[idx]
            push!(path, current)
        end
    
        reverse!(path)
        return path
    end

    function bfs_all_shortest_paths_within_shell(g::Graph, src::Int, dst::Int, C::Int)
        all_paths = []
        queue = [[src]]
        shortest_length = typemax(Int)

        while !isempty(queue)
            path = popfirst!(queue)
            current = last(path)

            if length(path)-1 > C
                continue
            end

            if current == dst
                if length(path) - 1 < shortest_length
                    shortest_length = length(path) - 1
                    empty!(all_paths)
                    push!(all_paths, path)
                elseif length(path) - 1 == shortest_length
                    push!(all_paths, path)
                end

                continue
            end
            # if current == dst
            #     push!(all_paths, path)
            #     continue
            # end

            for neighbor in neighbors(g, current)
                if neighbor ∉ path
                    new_path = copy(path)
                    push!(new_path, neighbor)
                    push!(queue, new_path)
                end
            end
        end

        return all_paths
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
                push!(edge_list, (u < v ? (u, v) : (v, u)))
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
end





 
# Create a random graph
N = 10000
k = 4
m = Int(N * k / 2)
G_ER = erdos_renyi(N, m)
C = Int[1,3,4,5]

# benchmark for single simulation
for c in C
    @info "Running simulation for C = $c"
    OD_pairs = find_node_pairs_within_distance(G_ER, c)
    b = @benchmark run_single_spp($G_ER, $c, $OD_pairs)
    display(b)
end

# Time complexity analysis for C=3
Ns = 10 .^[2,3,4]
times = Float64[]

for N in Ns
    k = 4
    m = Int(N * k / 2)
    G = erdos_renyi(N, m)
    OD_pairs = find_node_pairs_within_distance(G, 3)
    
    b = @benchmark run_single_spp_newman($G, 3, $OD_pairs)
    push!(times, median(b).time / 1e9)  # Convert to seconds
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

















num_simulations = 100
windows = round.(Int,collect(range(1, ne(G_ER)-1;length=50)))
p_values = windows./(ne(G_ER))
S_loop = zeros(Float64, num_simulations, length(windows));
χ_loop = zeros(Float64, num_simulations, length(windows));

h1 = plot(title="Susceptibility ",
    xlabel="p", ylabel="χ", legend=:topleft, grid=true, size=(800, 600));

h2 = plot(title=" Size of the Giant Component",
    xlabel="p", ylabel="S", legend=:topleft, grid=true, size=(800, 600));


for c in C
    @info "Running simulation for C = $c"
    OD_pairs = find_node_pairs_within_distance(G_ER, c)
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
    plot!(h2,1 .- p_values, S, label="(C=$c)", xlabel="p", ylabel="S")
    plot!(h1,1 .- p_values, χ, label="(C=$c)")
end


display(h1)
display(h2)
plot!(h1, 
    xguidefontsize=14, 
    yguidefontsize=14,
    legendfontsize=12,
    xtickfontsize=12,
    ytickfontsize=12,dpi=600)

plot!(h2,
    xguidefontsize=14,
    yguidefontsize=14, 
    legendfontsize=12,
    xtickfontsize=12,
    ytickfontsize=12,dpi=600)
savefig(h1, "Figure/SSP/Susceptibility.png")
savefig(h2, "Figure/SSP/Giant_component.png")

S_loop









## Parallel version
# Create a random graph
N = 100_00
k = 4
m = Int(N * k / 2)
G_ER = erdos_renyi(N, m)
C = 4
OD_pairs = find_node_pairs_within_distance(G_ER, C)
num_trials = 100
trial_indices = 1:num_trials

results = @showprogress pmap(trial_indices) do trial
    run_single_spp_newman(G_ER, C, OD_pairs)
end;

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
@info "Success with $nmax_cpu workers! For N=$N, estimated percolation threshold p_c ≈ $(round(p_c_est, digits=4))"



h3 = plot(title="Susceptibility ",
    xlabel="p", ylabel="χ", legend=:topleft, grid=true, size=(800, 600));
plot!(h3,1 .- p_vals, chi_small, label="(C=$C)", xlabel="p", ylabel="χ")
h4 = plot(title=" Size of the Giant Component",
    xlabel="p", ylabel="S", legend=:topleft, grid=true, size=(800, 600));
plot!(h4,1 .- p_vals, s_max_avg, label="(C=$C)", xlabel="p", ylabel="S")


savefig(h3, "Susceptibility_C=$C.png")
savefig(h4, "Giant_component_C=$C.png")