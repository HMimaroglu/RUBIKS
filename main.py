from rubiks import Rubiks

def main():
    r1 = Rubiks()
    print("Original solved state:")
    print(r1.faces)
    
    print("\nScrambling cube with 3 moves...")
    r1.randomize(100)
    print("Scrambled state:")
    print(r1.faces)
    
    print("\nSolving with BFS...")
    moves_to_solve = r1.solve_DFS()
    print(f"Solution found in {moves_to_solve} moves")
    
    print("\nNote: Original cube is still scrambled. BFS finds solution without modifying the cube.")

  
    
    

if __name__ == "__main__":
    main()