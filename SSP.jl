using Distributed
nmax_cpu = Sys.CPU_THREADS
if nprocs() == 1
    addprocs(nmax_cpu)
end


@everywhere begin
    using Plots
    using ProgressMeter
    using BenchmarkTools
    using Graphs
    using Random, Distributions
    using StatsBase
end

@everywhere include("MySPP.jl")
@everywhere using .MySPP
@everywhere using .Newman_Ziff

C_loop = [1,3,6,10]
k = Int(4)
for C in C_loop
N = 100000
g = erdos_renyi(N, Int(k * N / 2))


N = nv(g)
# build adj+eid once
adj = build_indexed_adjacency(g)

# one‐time state allocation
st = SPPState(
    Vector{Int}(undef, N),
    Vector{Int}(undef, N),
    [Int[] for _ in 1:N],
    Vector{Int}(undef, N + 1),
    adj,
    falses(ne(g)),
    Vector{Int64}(),
)

# Profile a single run
od_pairs = find_node_pairs_within_distance(g, C)

num_trials = 500
# Wrap pmap with @showprogress to display progress.
trial_indices = 1:num_trials
results = @showprogress pmap(trial_indices) do trial
    run_single_trial_spp_newman(g, C, od_pairs, st)
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

s_max_sq_mean = vec(mean(s_max_mat .^ 2, dims=1))
chi_giant = (s_max_sq_mean ./ N^2 .- GCC_frac .^ 2) ./ GCC_frac

idx = argmax(chi_small)
p_c_est = p_vals[idx]
@info "Success with $(nprocs()) workers! For N=$N and C=$C, estimated percolation threshold p_c ≈ $(round(p_c_est, digits=4))"

end


using Plots
plot(1 .- p_vals,chi_small)
plot(1 .- p_vals, GCC_frac)




























C = [1, 2, 3, 10, 20, 40, 80, 1500]
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
    yscale=:log10, dpi=600)
# Fit and plot trend line
log_C = log.(C)
log_t = log.(times)
slope = (length(C) * sum(log_C .* log_t) - sum(log_C) * sum(log_t)) /
        (length(C) * sum(log_C .^ 2) - sum(log_C)^2)

@info "Empirical complexity: O(N^$(round(slope, digits=2)))"
savefig(p, "time_complexityvsC.png")


Ns = 10 .^ [2, 3, 4]
C = 3
times = Float64[]
for n in Ns
    m = Int(n * 4 / 2)
    G = erdos_renyi(n, m)
    OD_pairs = find_node_pairs_within_distance(G, C)
    b = @benchmark run_single_spp_newman($G, $C, $OD_pairs)
    @info "Generating graph with N=$n, minimum time: $(minimum(b).time / 1e9)"
    push!(times, minimum(b).time / 1e9)
end


log_N = log.(Ns)
log_t = log.(times)
slope = (length(Ns) * sum(log_N .* log_t) - sum(log_N) * sum(log_t)) /
        (length(Ns) * sum(log_N .^ 2) - sum(log_N)^2)


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

savefig(p, "Figure/SSP/time_complexity.png")











N = 1000
C = 3
G_ER = erdos_renyi(N, Int(4 * N / 2))
num_simulations = 100
windows = round.(Int, collect(range(1, ne(G_ER) - 1; length=50)))
p_values = windows ./ (ne(G_ER))
S_loop = zeros(Float64, num_simulations, length(windows));
χ_loop = zeros(Float64, num_simulations, length(windows));

h1 = plot(title="Susceptibility ",
    xlabel="p", ylabel="χ", legend=:topleft, grid=true, size=(800, 600),
    xguidefontsize=14,
    yguidefontsize=14,
    legendfontsize=12,
    xtickfontsize=12,
    ytickfontsize=12, dpi=600);

h2 = plot(title=" Size of the Giant Component",
    xlabel="p", ylabel="S", legend=:topleft, grid=true, size=(800, 600),
    xguidefontsize=14,
    yguidefontsize=14,
    legendfontsize=12,
    xtickfontsize=12,
    ytickfontsize=12, dpi=600);



@info "Running simulation for C = $c"
OD_pairs = find_node_pairs_within_distance(G_ER, C)
@showprogress for i in 1:num_simulations
    t, edge_list_tau = run_single_spp(G_ER, c, OD_pairs)
    edge_list = [(x...,) for ee in edge_list_tau for x in ee]
    reverse!(edge_list)
    _, s_max_trial, chi_trial = run_single_trial(nv(G_ER), edge_list, windows, shuffle=false)
    S_loop[i, :] = s_max_trial
    χ_loop[i, :] = chi_trial
end
S = vec(mean(S_loop, dims=1)) / nv(G_ER)
χ = vec(mean(χ_loop, dims=1))
plot!(h2, 1 .- p_values, S, label="(C=$C) (no weights)", xlabel="p", ylabel="S")
plot!(h1, 1 .- p_values, χ, label="(C=$C) (no weights)")



display(h1)
display(h2)