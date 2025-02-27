using Random
using Statistics

"""
    generate_lattice(L, p)

Create an L×L Boolean matrix where each site is occupied (true) 
with probability p (site percolation).
"""
function generate_lattice(L::Int, p::Float64)
    lat = falses(L, L)
    for i in 1:L, j in 1:L
        lat[i, j] = (rand() < p)
    end
    return lat
end
    
"""
    find_clusters(lat)

Given a Boolean L×L matrix `lat`, returns a Dict(cluster_id => cluster_size).
"""
function find_clusters(lat::BitMatrix)
    L = size(lat, 1)
    visited = falses(L, L)
    cluster_id = fill(0, L, L)
    current_id = 0
    cluster_sizes = Dict{Int, Int}()

    # 4-neighborhood (up, down, left, right)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for i in 1:L, j in 1:L
        if lat[i, j] && !visited[i, j]
            current_id += 1
            queue = [(i, j)]
            visited[i, j] = true
            cluster_id[i, j] = current_id
            size_count = 0

            # BFS to find all connected sites
            while !isempty(queue)
                (x, y) = pop!(queue)
                size_count += 1

                for (dx, dy) in directions
                    nx, ny = x + dx, y + dy
                    if 1 ≤ nx ≤ L && 1 ≤ ny ≤ L
                        if lat[nx, ny] && !visited[nx, ny]
                            visited[nx, ny] = true
                            cluster_id[nx, ny] = current_id
                            push!(queue, (nx, ny))
                        end
                    end
                end
            end

            cluster_sizes[current_id] = size_count
        end
    end

    return cluster_sizes
end 

"""
    calc_susceptibility(cluster_sizes)

Given a Dict(cluster_id => size), compute the 
percolation "susceptibility": sum(s^2 * n_s) / sum(s * n_s).
"""
function calc_susceptibility(cluster_sizes::Dict{Int,Int})
    size_count = collect(values(cluster_sizes))
    size_count = size_count[size_count .!= maximum(size_count)]
    chi = sum(size_count.^2) / sum(size_count)
    return chi 
end

"""
    check_percolation(lat, cluster_id)
    Check if there's a percolating cluster (spanning from top to bottom or left to right).
"""
function check_percolation(lat::BitMatrix, cluster_id::Matrix{Int})
    L = size(lat, 1)
    
    # Check vertical percolation
    top_clusters = Set(cluster_id[1, :])
    bottom_clusters = Set(cluster_id[L, :])
    vertical = any(id -> id != 0 && id in bottom_clusters, top_clusters)
    
    # Check horizontal percolation
    left_clusters = Set(cluster_id[:, 1])
    right_clusters = Set(cluster_id[:, L])
    horizontal = any(id -> id != 0 && id in right_clusters, left_clusters)
    
    return vertical || horizontal
end

"""
    simulate_once(L, p)

Generate a random L×L lattice at probability p, 
find clusters, return the largest cluster size, susceptibility, and percolation status.
"""
function simulate_once(L::Int, p::Float64)
    lat = generate_lattice(L, p)
    clusters = find_clusters(lat)
    cluster_id = fill(0, L, L)
    # Need to reconstruct cluster_id matrix
    for i in 1:L, j in 1:L
        if lat[i,j]
            for (id, _) in clusters
                if cluster_id[i,j] == 0 # not yet assigned
                    cluster_id[i,j] = id
                end
            end
        end
    end
    percolates = check_percolation(lat, cluster_id)
    return maximum(values(clusters)), calc_susceptibility(clusters), percolates
end

"""
    simulate(L, p, Nsamples)

Run Nsamples independent simulations for a given (L, p).
Return the mean and std of susceptibility, largest cluster size, and percolation probability.
"""
function simulate(L::Int, p::Float64; Nsamples::Int=10)
    sus_vals = Float64[]
    smax_vals = Float64[]
    perc_vals = Float64[]
    for _ in 1:Nsamples
        smax, sus, perc = simulate_once(L, p)
        push!(sus_vals, sus)
        push!(smax_vals, smax)
        push!(perc_vals, Float64(perc))
    end
    perc_prob = mean(perc_vals)
    return (mean(sus_vals), std(sus_vals)), 
           (mean(smax_vals), std(smax_vals)), 
           perc_prob
end

# Example usage:
Ls = [128,256]         # Lattice sizes
ps = 0.50:0.01:0.65       # Probability range around expected p_c (~0.5927 in 2D)
Nsamples = 1000            # Number of realizations per (L, p)

# Dictionary to store results: key = (L, p), value = (mean_susceptibility, std_susceptibility)
results = Dict{Tuple{Int,Float64}, Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Float64}}()

for L in Ls
    for p in ps
        results[(L, p)] = simulate(L, p; Nsamples=Nsamples)
        # @show L, p, m, s  # Print or log results
    end
end


# 1) Plot m (susceptibility) vs. p for each L (error bars = std)
using Plots
h1 = plot(title="Percolation Susceptibility", xlabel="p", ylabel="Susceptibility");
h2 = plot(title="Percolation", xlabel="p", ylabel="Order Parameter");
h3 = plot(title="Percolation", xlabel="p", ylabel="Probability (One side)");
markers = [:circle, :square, :diamond, :utrangle]
for (idx, L) in enumerate(Ls)
    chi_m = [results[(L, p)][1][1] for p in ps]
    chi_std = [results[(L, p)][1][2] for p in ps]
    S_m = [results[(L, p)][2][1] for p in ps]
    S_std = [results[(L, p)][2][2] for p in ps]
    perc_prob = [results[(L, p)][3] for p in ps]    
    plot!(h1, ps, chi_m, label="L = $L",
          markershape=markers[idx], markercolor=:white,
          markerstrokewidth=2)
    plot!(h2, ps, S_m/L^2, label="L = $L", 
          markershape=markers[idx], markercolor=:white,
          markerstrokewidth=2)
    plot!(h3, ps, perc_prob, label="L = $L", 
          markershape=markers[idx], markercolor=:white,
          markerstrokewidth=2)
end

display(plot(h1, h2,layout=(2,1),size=(1200,800)))
