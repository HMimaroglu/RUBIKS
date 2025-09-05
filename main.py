
from rubiks import Rubiks
def main():
    r1 = Rubiks()
    r1.randomize()
    
    print("Attempting to solve with AI...")
    solution = r1.solve_ai()
    
    
    


if __name__ == "__main__":
    main()