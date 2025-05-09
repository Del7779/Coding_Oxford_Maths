using Plots, Random

# ─────────────── Simulation helpers ───────────────

# 2D Lévy–flight
function levy_flight_2d(N; α=1.5, ℓ0=1.0)
    θ = 2π .* rand(N)
    # Pareto(ℓ0,α) via inverse‐CDF: ℓ = ℓ0 * (1 - U)^(-1/α)
    U = rand(N)
    ℓ = ℓ0 .* (1 .- U).^(-1/α)
    steps = vcat((ℓ .* cos.(θ))',(ℓ .* sin.(θ))')
    return cumsum(hcat(zeros(2), steps), dims=2)
end

# 2D Gaussian random walk
function random_walk_2d(N; σ=1.0)
    steps = σ .* randn((2, N))
    return cumsum(hcat(zeros(2), steps), dims=2)
end

# ─────────────── Build trajectories ───────────────

N = 500
lf = levy_flight_2d(N; α=1.3, ℓ0=1.0)
rw = random_walk_2d(N; σ=1.0)

# ─────────────── Create animation ───────────────

# Use GR backend
gr()

anim = @animate for i in 1:N+1
    # Lévy‐flight
    p1 = scatter(lf[1,1:i], lf[2,1:i];
                 xlim=(-50,50), ylim=(-50,50),
                 markersize=3, label="", title="Lévy flight")
    scatter!(p1, [lf[1,i]], [lf[2,i]]; markersize=6, c=:red, label="")

    # Gaussian RW
    p2 = scatter(rw[1,1:i], rw[2,1:i];
                 xlim=(-50,50), ylim=(-50,50),
                 markersize=3, label="", title="Gaussian RW")
    scatter!(p2, [rw[1,i]], [rw[2,i]]; markersize=6, c=:blue, label="")

    plot(p1, p2; layout=(1,2), size=(800,400))
end

gif(anim, "levy_vs_rw.gif", fps=15)
