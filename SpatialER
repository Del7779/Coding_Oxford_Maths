using Distributions, Random, Plots, StatsBase, LinearAlgebra

# Parameters
N = 10     # Number of nodes
λ = 5                # Mean degree (Poisson distribution)
alpha = -5          # Connection probability ~ distance^alpha
L = 100.0           # Domain size (L x L)

# Generate Poisson-distributed degrees
degree_dist = Poisson(λ)
degrees = rand(degree_dist, N)
degree_remaining = copy(degrees)  # Track remaining degrees to allocate

# Generate random positions
x = rand(N) * L
y = rand(N) * L

scatter(x, y, markersize=degrees, color=:red, alpha=0.7, label="", 
    markerstrokecolor=:black, markerstrokewidth=0.5, title="Network")

# Compute pairwise distances
distances = [sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2) for i in 1:N, j in 1:N]
# distances[tril!(trues(N, N))] .= NaN  # Exclude diagonal and lower triangle
distances[diagind(distances)] .= NaN 
# Normalized probability function
function connection_prob(d, degree_remaining, alpha)
    indices = findall(x -> !isnan(x) && x != 0, vec(d))  # Find valid indices
    Z = sum(degree_remaining[indices] .* (d[indices] .^ alpha))
    probs = (d .^ alpha) ./ Z .* degree_remaining
    probs[isnan.(probs) .| (probs .== 0)] .= 0  # Replace NaN and 0 values with 0
    return probs
end


# Generate edges based on probability and degree constraints
edges = []
# stubs = [degree_remaining[i] for i in 1:N]
stubs = [i for i in 1:N for _ in 1:degree_remaining[i]]  # Create a list of stubs
  # Create a list of stubs
shuffle!(stubs)  # Shuffle the stubs to ensure randomness

while length(stubs) > 1
    i = pop!(stubs)  # Pick a stub (node i)
    probs = connection_prob(distances[i, :], degree_remaining, alpha)
    # Choose j according to probability Π (excluding i itself)
    j = sample(1:N, Weights(probs))
    
    if i != j && (i, j) ∉ edges && (j, i) ∉ edges  # Avoid self-loops and duplicate edges
        push!(edges, (i, j))
        degree_remaining[i] -= 1
        degree_remaining[j] -= 1
    else
        push!(stubs, i)  # Reinsert the stub if no valid connection is found
    end
    # plot(probs)
    # plot(degree_remaining)
end

plt_network = plot()
pyplot
# Create the network plot (scatter nodes)
scatter!(plt_network, x, y, markersize=degrees, color=:red, alpha=0.7, label="", 
    markerstrokecolor=:black, markerstrokewidth=0.5, title="Network")
# Draw edges on the same figure
for (i, j) in edges
    plot!(plt_network, [x[i], x[j]], [y[i], y[j]], color=:black, lw=0.5, label="", alpha=0.1)
end
display(plt_network)




# Create the histogram
pkg"add StatsPlots"
using StatsPlots
# Plot probability density function (PDF) curve
plt_hist = plot()
density!(plt_hist, [distances[i, j] for (i, j) in edges], linewidth=2, color=:blue, label="Probability Density",
    xlabel="Link Length", ylabel="Density", title="Link Length Probability Density Curve")
layout = @layout [a b]
# Combine plots
plt1 = plot(plt_network, plt_hist, layout=layout, size=(900,400))
display(plt1)


d_test = range(minimum(skipmissing(distances[:])), maximum(skipmissing(distances[:])), length=100)
pp_test = [connection_prob(d, degree_remaining, alpha) for d in d_test]
plot(d_test,pp_test,xscale=:log10,yscale=:log10,legend=false,xlabel="Distance",ylabel="Connection Probability",title="Connection Probability vs. Distance")
