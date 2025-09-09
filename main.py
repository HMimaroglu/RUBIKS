from rubiks import Rubiks

def main():
    r1 = Rubiks()
    print(r1.faces)
    r1.randomize(1000)
    
    
    print(r1.solve_BFS())
    print(r1.faces)

  
    
    

if __name__ == "__main__":
    main()