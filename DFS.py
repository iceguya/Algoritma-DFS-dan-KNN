import math

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return f"<Node {self.state}>"

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        new_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
        return Node(next_state, self, action, new_cost)

    def solution(self):
        path = []
        node = self
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1] 

class Problem:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def path_cost(self, cost_so_far, state1, action, state2):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal_state

# --- Implementasi Khusus untuk Peta Rumania ---
class RomaniaMap(Problem):
    def __init__(self, initial_city, goal_city):
        super().__init__(initial_city, goal_city)
        self.graph = {
            'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
            'Zerind': {'Arad': 75, 'Oradea': 71},
            'Oradea': {'Zerind': 71, 'Sibiu': 151},
            'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
            'Timisoara': {'Arad': 118, 'Lugoj': 111},
            'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
            'Mehadia': {'Lugoj': 70, 'Dobreta': 75},
            'Dobreta': {'Mehadia': 75, 'Craiova': 120},
            'Craiova': {'Dobreta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
            'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
            'Fagaras': {'Sibiu': 99, 'Bucuresti': 211},
            'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucuresti': 101},
            'Bucuresti': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
            'Giurgiu': {'Bucuresti': 90},
            'Urziceni': {'Bucuresti': 85, 'Hirsova': 98, 'Vaslui': 142},
            'Hirsova': {'Urziceni': 98, 'Eforie': 86},
            'Eforie': {'Hirsova': 86},
            'Vaslui': {'Urziceni': 142, 'Iasi': 92},
            'Iasi': {'Vaslui': 92, 'Neamt': 87},
            'Neamt': {'Iasi': 87}
        }

    def actions(self, city):
        return list(self.graph.get(city, {}).keys())

    def result(self, current_city, next_city):
        if next_city in self.graph.get(current_city, {}):
            return next_city
        raise ValueError(f"Invalid transition from {current_city} to {next_city}")

    def path_cost(self, cost_so_far, city1, action, city2):
        return cost_so_far + self.graph[city1][city2]

# --- Algoritma Depth-First Search (DFS) ---
def depth_first_search(problem):
    fringe = [Node(problem.initial_state)]
    explored = set()
    while fringe:
        node = fringe.pop() 
        if problem.goal_test(node.state):
            return node.solution() 
        if node.state not in explored:
            explored.add(node.state)
            for child in reversed(node.expand(problem)):
                if child.state not in explored and child not in (n.state for n in fringe):
                    fringe.append(child)
    return None 


if __name__ == "__main__":
    print("Program Pencarian Jalur di Peta Rumania Menggunakan Depth-First Search (DFS)\n")

    # Skenario 1: Arad ke Bucuresti (Ini adalah contoh klasik di buku)
    initial_city_1 = 'Arad'
    goal_city_1 = 'Bucuresti'
    romania_problem_1 = RomaniaMap(initial_city_1, goal_city_1)
    print(f"Mencari jalur dari {initial_city_1} ke {goal_city_1}...")
    solution_path_1 = depth_first_search(romania_problem_1)

    if solution_path_1:
        print("Jalur ditemukan:", " -> ".join(solution_path_1))
        # Hitung biaya path solusi yang ditemukan (ini adalah biaya dari jalur DFS)
        current_cost = 0
        for i in range(len(solution_path_1) - 1):
            city_a = solution_path_1[i]
            city_b = solution_path_1[i+1]
            current_cost += romania_problem_1.graph[city_a][city_b]
        print(f"Total biaya (jarak) jalur yang ditemukan: {current_cost} km\n")
    else:
        print(f"Tidak ada jalur yang ditemukan dari {initial_city_1} ke {goal_city_1}.\n")

    # Skenario 3: Mencoba mencari jalur ke kota yang tidak terhubung (contoh)
    initial_city_3 = 'Arad'
    goal_city_3 = 'London' # Kota yang tidak ada di peta Rumania
    romania_problem_3 = RomaniaMap(initial_city_3, goal_city_3)
    print(f"Mencari jalur dari {initial_city_3} ke {goal_city_3}...")
    solution_path_3 = depth_first_search(romania_problem_3)

    if solution_path_3:
        print("Jalur ditemukan:", " -> ".join(solution_path_3))
        current_cost = 0
        for i in range(len(solution_path_3) - 1):
            city_a = solution_path_3[i]
            city_b = solution_path_3[i+1]
            current_cost += romania_problem_3.graph[city_a][city_b]
        print(f"Total biaya (jarak) jalur yang ditemukan: {current_cost} km\n")
    else:
        print(f"Tidak ada jalur yang ditemukan dari {initial_city_3} ke {goal_city_3}.\n")
