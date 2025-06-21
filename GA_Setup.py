import random
import matplotlib.pyplot as plt
import numpy as np

# Game setup
BOARD_SIZE = (3, 9)
TOTAL_CELLS = BOARD_SIZE[0] * BOARD_SIZE[1]
PIECES = [
    '5SG', '4SG', '3SG', '2SG', '1SG', 'COL', 'LTC', 'MAJ', 'CPT',
    '1LT', '2LT', 'SRG', 'PVT1', 'PVT2', 'PVT3', 'PVT4', 'PVT5', 'PVT6',
    'SPY1', 'SPY2', 'FLG'
]

def flatten_board():
    return list(range(TOTAL_CELLS))

FLAG_POSITION = 13  # Fixed for consistency

# Chromosome generator
def generate_chromosome():
    available_positions = list(flatten_board())
    available_positions.remove(FLAG_POSITION)
    random.shuffle(available_positions)

    chrom = []
    for piece in PIECES:
        if piece == 'FLG':
            chrom.append((piece, FLAG_POSITION))
        else:
            chrom.append((piece, available_positions.pop()))
    return chrom

def assert_chromosome_valid(chrom):
    assert sorted(p for p, _ in chrom) == sorted(PIECES), "❌ Invalid chromosome: Missing or duplicate pieces"

def assert_no_position_conflicts(chrom):
    positions = [pos for _, pos in chrom]
    assert len(set(positions)) == len(positions), "❌ Overlapping positions detected!"

# Repair chromosome to avoid duplicate positions
def repair_chromosome(chrom):
    seen = {}
    for i, (piece, pos) in enumerate(chrom):
        if pos not in seen:
            seen[pos] = i
        else:
            # Replace with a new available position
            used_positions = set(seen.keys())
            used_positions.add(FLAG_POSITION)  # Don't overwrite flag
            all_positions = set(flatten_board())
            available = list(all_positions - used_positions)
            if available:
                new_pos = random.choice(available)
                chrom[i] = (piece, new_pos)
                seen[new_pos] = i
    return chrom

# Fitness heuristics
def evaluate_proximity(chrom):
    flag_pos = next(pos for piece, pos in chrom if piece == 'FLG')
    defenders = [pos for piece, pos in chrom if piece in {'5SG', '4SG', '3SG', '2SG', '1SG'}]
    return np.mean([1 / (1 + abs(p - flag_pos)) for p in defenders]) if defenders else 0

def evaluate_deception(chrom):
    positions = [pos for piece, pos in chrom if piece.startswith('PVT') or piece.startswith('SPY')]
    return np.std(positions) / TOTAL_CELLS if positions else 0

def evaluate_cohesion(chrom):
    group = [pos for piece, pos in chrom if piece in {'GEN', 'COL', 'MAJ'}]
    return 1 / (1 + np.std(group)) if group else 0

def evaluate_balance(chrom):
    positions = [pos % BOARD_SIZE[1] for _, pos in chrom]
    return 1 - (np.std(positions) / BOARD_SIZE[1]) if positions else 0

def penalize_clustering(chrom):
    high_value = [pos for piece, pos in chrom if piece in {'5SG', '4SG', 'COL'}]
    return sum([abs(p1 - p2) < 2 for i, p1 in enumerate(high_value) for p2 in high_value[i+1:]]) / len(high_value) if high_value else 0

def fitness(chrom):
    return (
        0.25 * evaluate_proximity(chrom) +
        0.20 * evaluate_deception(chrom) +
        0.20 * evaluate_cohesion(chrom) +
        0.25 * evaluate_balance(chrom) -
        0.10 * penalize_clustering(chrom)
    )

# Genetic operators
def tournament_selection(pop, k=3):
    return max(random.sample(pop, k), key=fitness)

def two_point_crossover(p1, p2):
    p1_map = {piece: pos for piece, pos in p1}
    p2_map = {piece: pos for piece, pos in p2}
    cut1, cut2 = sorted(random.sample(range(len(PIECES)), 2))
    child = []
    for i, piece in enumerate(PIECES):
        pos = p2_map[piece] if cut1 <= i < cut2 else p1_map[piece]
        child.append((piece, pos))
    return child

def mutate(chrom, rate=0.1):
    if random.random() < rate:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = (chrom[j][0], chrom[i][1]), (chrom[i][0], chrom[j][1])
    return chrom

def avg_hamming(pop):
    def hamming(c1, c2):
        return sum(pos1 != pos2 for (_, pos1), (_, pos2) in zip(c1, c2))
    return np.mean([hamming(pop[i], pop[j]) for i in range(len(pop)) for j in range(i+1, len(pop))])

# GA Main Loop
# GA Main Loop
POP_SIZE = 100
GENERATIONS = 50
population = [generate_chromosome() for _ in range(POP_SIZE)]
best_scores, avg_scores, diversity_scores = [], [], []

for gen in range(GENERATIONS):
    next_gen = []
    for _ in range(POP_SIZE):
        p1, p2 = tournament_selection(population), tournament_selection(population)
        child = two_point_crossover(p1, p2)
        child = mutate(child)
        child = repair_chromosome(child)
        assert_chromosome_valid(child)
        next_gen.append(child)

    for chrom in population:
        assert_chromosome_valid(chrom)
        assert_no_position_conflicts(chrom)

    population = next_gen
    best = max(population, key=fitness)
    best_fit = fitness(best)
    avg_fit = np.mean([fitness(c) for c in population])
    diversity = avg_hamming(population)

    best_scores.append(best_fit)
    avg_scores.append(avg_fit)
    diversity_scores.append(diversity)

    # 🔍 Print fitness score each generation
    print(f"Generation {gen + 1}: Best = {best_fit:.4f}, Avg = {avg_fit:.4f}")

# Display Best Chromosome
best_chrom = max(population, key=fitness)
print("\n🎯 Best Chromosome Arrangement:")
print(f"Fitness Score: {fitness(best_chrom):.4f}\n")

# Display as board
board_view = ['[---]' for _ in range(TOTAL_CELLS)]
for piece, pos in best_chrom:
    board_view[pos] = f"[{piece}]"

rows = "ABC"
for i in range(BOARD_SIZE[0]):
    print(f"{rows[i]} " + " ".join(board_view[i * BOARD_SIZE[1]:(i + 1) * BOARD_SIZE[1]]))
print("   " + "   ".join(str(i) for i in range(1, BOARD_SIZE[1] + 1)))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(best_scores, label='Best Fitness')
plt.plot(avg_scores, label='Average Fitness')
plt.plot(diversity_scores, label='Diversity')
plt.title('GA Convergence and Diversity')
plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
