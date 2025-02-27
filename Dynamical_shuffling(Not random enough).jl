using StatsBase  # For the sample() function
# -------------------------------
# Utility functions for the shuffle
# -------------------------------

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
function shuffled_edges(N::Int)
    T = div(N * (N - 1), 2)  # total number of edges in a complete graph on N nodes
    a = random_coprime(T)   # choose a random multiplier that is coprime with T
    b = rand(0:T-1)         # choose an arbitrary offset
    # The mapping r -> mod(a*r + b, T) defines a permutation of 0:(T-1).
    return ( edge_unrank(mod(a * r + b, T), N) for r in 0:(T-1) )
end