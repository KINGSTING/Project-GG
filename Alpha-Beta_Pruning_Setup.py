import random


def get_random_position():
    """Return a random position on the grid in the format 'A3'."""
    # Define the valid rows and columns
    rows = ['A', 'B', 'C']
    columns = [str(i) for i in range(1, 10)]  # Columns 1 to 9

    # Select random row and column
    random_row = random.choice(rows)
    random_col = random.choice(columns)

    # Combine row and column into the format 'A3'
    return random_row + random_col


class GeneticGame:
    def __init__(self, population_size, generations, tournament_size):
        self.flag_position = get_random_position()
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size

        # Define pieces (unique entries, no duplicates)
        self.pieces = [
            '5SG', '4SG', '3SG', '2SG', '1SG', 'COL', 'LTC', 'MAJ', 'CPT',
            '1LT', '2LT', 'SRG', 'PVT1', 'PVT2', 'PVT3', 'PVT4', 'PVT5', 'PVT6', 'SPY1', 'SPY2', 'FLG'
        ]

        # Update to 3 rows (A, B, C) and 9 columns (1 to 9)
        self.player_slots = [f"{row}{col}" for row in "ABC" for col in range(1, 10)]  # 3 rows and 9 columns -> 27 slots
        self.mid_tier_slots = [f"B{col}" for col in range(4, 10)]  # B4 to B9 for mid-tier officers

    def generate_random_chromosome(self):
        """Generate a valid random chromosome (pieces randomly assigned to slots), with FLG in a fixed position."""
        # Shuffle the list of pieces, excluding 'FLG' for fixed position
        pieces_without_flg = [piece for piece in self.pieces if piece != 'FLG']
        shuffled_pieces = pieces_without_flg.copy()
        random.shuffle(shuffled_pieces)  # Shuffle the pieces to make them random

        # Shuffle all available slots
        all_slots = self.player_slots.copy()
        random.shuffle(all_slots)  # Shuffle all slots to make the assignment random

        # Assign the FLG to a fixed position (this could be set somewhere else in the class)
        flg_position = self.flag_position if self.flag_position else random.choice(self.player_slots)

        # Remove the FLG position from the list of available slots
        all_slots.remove(flg_position)

        # Combine shuffled pieces with shuffled slots and ensure FLG is placed in the fixed position
        chromosome = [(piece, slot) for piece, slot in zip(shuffled_pieces, all_slots)]
        chromosome.append(('FLG', flg_position))  # Append FLG in its fixed position

        return chromosome

    @staticmethod
    def update(chromosome):
        """Display the board setup based on the given chromosome."""
        board = [['[---]' for _ in range(9)] for _ in range(3)]  # 4 rows for player 1
        for piece, slot in chromosome:
            row = ord(slot[0]) - ord('A')
            col = int(slot[1]) - 1
            board[row][col] = f"[{piece}]"
        print("\nCurrent Board Setup:")
        for row, row_data in zip("ABC", board):
            print(f"{row} " + " ".join(row_data))
        print("    " + "   ".join(map(str, range(1, 10))) + "\n")

    @staticmethod
    def fitness(chromosome):
        """Evaluate the fitness of a chromosome based on Clustered Task Force strategy with specific group structure."""
        max_possible_score = 200  # Adjusted max score for Clustered Task Forces strategy
        score = 0

        # Define a dictionary to track the positions of key pieces for the task force
        task_force_pieces = ['5SG', '4SG', 'SPY1', 'SPY2', 'PVT1', 'PVT2', 'COL', 'LTC', 'MAJ', 'CPT']
        task_force_proximity_bonus = 10  # Bonus for placing the task force together
        movement_penalty = -5  # Penalize for non-movement of key pieces

        # Parameters for scoring
        for piece, slot in chromosome:
            row, col = slot[0], int(slot[1])

            # Task Force Grouping (with specific pieces)
            if piece in ['5SG', '4SG']:
                # Find the associated task force members
                nearby_units = sum(
                    1 for other_piece, other_slot in chromosome
                    if other_piece in task_force_pieces and abs(ord(row) - ord(other_slot[0])) <= 1 and abs(
                        col - int(other_slot[1])) <= 1
                )
                if nearby_units >= 5:  # Reward the proximity of 5 pieces (General + Spy + 2 Privates + 2 Officers)
                    score += task_force_proximity_bonus

                # Encourage movement: Penalize if the 5SG or 4SG stays too still
                if row == 'B' and col in [4, 5,
                                          6]:  # High-ranking pieces like Spies and Officers should not stay too static
                    score += movement_penalty

            # Privates (PVT)
            if piece.startswith('PVT'):
                # Reward placement near the high-ranking general or officers
                nearby_general_or_officer = sum(
                    1 for other_piece, other_slot in chromosome
                    if
                    (other_piece.endswith('SG') or other_piece in ['SPY1', 'SPY2', 'COL', 'LTC', 'MAJ', 'CPT']) and abs(
                        ord(row) - ord(other_slot[0])) <= 1 and abs(col - int(other_slot[1])) <= 1
                )
                score += 4 * nearby_general_or_officer

                # Reward for frontline placement (row C)
                if row == 'C':
                    score += 6
                elif row == 'B':  # Second row for support
                    score += 4
                else:  # Backline placement (less effective for privates)
                    score -= 2

            # Officers (COL, LTC, MAJ, CPT)
            if piece in ['COL', 'LTC', 'MAJ', 'CPT']:
                # Reward placement near high-ranking general or spy
                nearby_high_value = sum(
                    1 for other_piece, other_slot in chromosome
                    if other_piece.endswith('SG') or other_piece in ['SPY1', 'SPY2'] and abs(
                        ord(row) - ord(other_slot[0])) <= 1 and abs(col - int(other_slot[1])) <= 1
                )
                score += 2 * nearby_high_value

                # Favor second-row placement (row B) for officers
                if row == 'B':
                    score += 5
                elif row == 'C':
                    score += 3
                elif row == 'A':
                    score -= 3

            # Penalize pieces left too far from the action (encourage cohesion)
            if row == 'A' and piece not in ['FLG']:
                score -= 2  # Penalize for pieces being too far from the frontline

        # Normalize the score
        return score / max_possible_score * 100

    def evolve(self):
        population = [self.generate_random_chromosome() for _ in range(self.population_size)]
        best_chromosome = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            # Calculate fitness and probabilities
            fitness_scores = [self.fitness(chromosome) for chromosome in population]

            # Select parents using tournament selection
            selection_size = int(self.population_size * 0.7)  # Adjust selection rate as needed
            parents = []

            for _ in range(selection_size):
                # Randomly select 'tournament_size' individuals
                tournament = random.sample(population, self.tournament_size)
                # Find the individual with the best fitness in the tournament
                best_tournament = max(tournament, key=lambda chromosome: self.fitness(chromosome))
                parents.append(best_tournament)

            next_generation = []
            for i in range(0, len(parents), 2):
                try:
                    # Check if there's a second parent available (avoids IndexError)
                    if i + 1 < len(parents):
                        parent1, parent2 = parents[i], parents[i + 1]
                        child1, child2 = self.crossover(parent1, parent2)
                        next_generation.extend([self.fix_chromosome(child1), self.fix_chromosome(child2)])
                    else:
                        # Handle the case of an odd number of parents (e.g., skip or duplicate)
                        continue
                except ValueError:
                    # Skip invalid chromosomes
                    continue

            # Refill population if necessary
            if len(next_generation) < self.population_size:
                for _ in range(self.population_size - len(next_generation)):
                    next_generation.append(self.generate_random_chromosome())

            # Update population
            population = next_generation

            # Track best fitness
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_chromosome = population[fitness_scores.index(max_fitness)]

            print(f"Generation {generation + 1}: Best Fitness = {max_fitness}")
            self.update(best_chromosome)

        return best_chromosome

    def crossover(self, parent1, parent2):
        """Performs two-point crossover between two parents, preserving FLG."""
        # Extract the fixed "FLG" position
        flag1 = next((gene for gene in parent1 if gene[0] == 'FLG'), None)
        flag2 = next((gene for gene in parent2 if gene[0] == 'FLG'), None)

        # Remove the flag from both parents
        parent1_no_flag = [gene for gene in parent1 if gene[0] != 'FLG']
        parent2_no_flag = [gene for gene in parent2 if gene[0] != 'FLG']

        # Choose two random crossover points
        crossover_point1 = random.randint(0, len(parent1_no_flag) - 1)
        crossover_point2 = random.randint(crossover_point1, len(parent1_no_flag))

        # Perform crossover
        child1_genes = (
                parent1_no_flag[:crossover_point1]
                + parent2_no_flag[crossover_point1:crossover_point2]
                + parent1_no_flag[crossover_point2:]
        )
        child2_genes = (
                parent2_no_flag[:crossover_point1]
                + parent1_no_flag[crossover_point1:crossover_point2]
                + parent2_no_flag[crossover_point2:]
        )

        # Reinsert the flag position
        child1 = [flag1] + child1_genes
        child2 = [flag2] + child2_genes

        return child1, child2

    @staticmethod
    def hamming_distance(chromosome1, chromosome2):
        distance = 0
        for gene1, gene2 in zip(chromosome1, chromosome2):
            if gene1 != gene2:
                distance += 1
        return distance

    def calculate_population_diversity(self, population):
        diversity = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diversity += self.hamming_distance(population[i], population[j])
        return diversity / (len(population) * (len(population) - 1) / 2)

    def mutate_chromosome(self, chromosome):
        """Perform mutation on a chromosome but keep the flag fixed in its position."""
        # Exclude the flag piece from mutation
        non_flag_chromosome = [piece for piece in chromosome if piece[0] != 'FLG']

        # Perform mutation on the non-flag pieces
        random.shuffle(non_flag_chromosome)

        # Insert the flag back into its fixed position
        mutated_chromosome = [(piece, self.flag_position) if piece == 'FLG' else (piece, slot) for piece, slot in non_flag_chromosome]

        return mutated_chromosome

    def fix_chromosome(self, chromosome):
        """Ensure the chromosome has all unique pieces and valid slots, preserving FLG position."""
        # Identify the FLG gene
        flag_gene = next((gene for gene in chromosome if gene[0] == 'FLG'), None)

        # Step 1: Identify duplicates and missing pieces
        assigned_pieces = [piece for piece, slot in chromosome if piece != 'FLG']
        assigned_slots = [slot for piece, slot in chromosome if piece != 'FLG']

        duplicates = [piece for piece in set(assigned_pieces) if assigned_pieces.count(piece) > 1]
        missing_pieces = [piece for piece in self.pieces if piece not in assigned_pieces and piece != 'FLG']
        available_slots = [slot for slot in self.player_slots if slot not in assigned_slots]

        # Step 2: Remove duplicates
        cleaned_chromosome = [gene for gene in chromosome if gene[0] not in duplicates and gene[0] != 'FLG']
        seen_pieces = set(gene[0] for gene in cleaned_chromosome)

        # Step 3: Add missing pieces to available slots
        for piece, slot in zip(missing_pieces, available_slots):
            cleaned_chromosome.append((piece, slot))

        # Step 4: Add the FLG gene back in its original position
        if flag_gene:
            cleaned_chromosome.append(flag_gene)

        # Step 5: Ensure chromosome length matches the expected size
        if len(cleaned_chromosome) != len(self.pieces):
            raise ValueError("Fixing failed: mismatch in chromosome size!")

        return cleaned_chromosome


game = GeneticGame(population_size=1000, generations=50, tournament_size=10)
best_setup = game.evolve()
print("Optimal Deployment Found:")
game.update(best_setup)
