using Distributions

function bt_sampler(D::Matrix{Int})
    const K = size(D,1)                 # number of teams
    const w = sum(D,2)                  # w[i] is the number of games
                                        # team i has won
    const n = D + D'                    # n[i,j] is the number of
                                        # times team i and j have
                                        # played (n[i,j] = n[j,i])

    ## Prior on λ 
    a = 1
    b = 1
    
    λ = ones(Float64, K)

    ## we actually only care about the sums of Z, so we're storing
    ## that in Zs here
    Zs = zeros(Float64, K)

    while true
        ## update Z
        for j = 2:K, i = 1:(j-1)
            if n[i,j] > 0
                val = rand(Gamma(n[i,j], 1.0/(λ[i] + λ[j])))
                Zs[i] += val
                Zs[j] += val
            end
        end
        
        ## update λ
        for i = 1:K
            λ[i] = rand(Gamma(a + w[i], 1.0/(b + Zs[i])))
            Zs[i] = 0.0 # clear our Zs while we're at it
        end
        produce(λ)
    end
end

######

# Data obtained from https://www.spreadsheet-sports.com/2015-ncaa-basketball-game-data
data = readcsv("2015.csv")
# We only care about D1 games
data = data[data[:, 12] .== "Division 1", :]

teams = unique([data[2:end, 2], data[2:end, 5]])

id = Dict{String, Int}()

for (e, team) in enumerate(teams)
    id[team] = e
end

D = zeros(Int, length(id), length(id))

for i = 1:size(data, 1)
    team1 = data[i, 2]
    score1 = data[i, 4]
    
    team2 = data[i, 5]
    score2 = data[i, 6]
    
    if score1 > score2
        D[id[team1],id[team2]] += 1
    else
        D[id[team2],id[team1]] += 1
    end
end

srand(42) # for reproducibility

nBurn = 1000
nDraw = 1000

sampler = @task bt_sampler(D)

## Burn in
for i = 1:nBurn
    consume(sampler)
end

## Draw from the posterior and do what you will
for i = 1:nDraw
    λ = consume(sampler)
end
