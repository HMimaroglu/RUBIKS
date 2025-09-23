import random
from collections import deque

class Rubiks:

    ## Initializes a solved 2x2 rubik's cube
    def __init__(self):
        self.faces = {
            'front': ['G'] * 4, # Green
            'back': ['B'] * 4, # Blue
            'bottom': ['R'] * 4, # Red
            'right': ['W'] * 4, # White
            'left': ['Y'] * 4, # Yellow
            'top': ['O'] * 4  # Orange
        }

        self.move_options = ['ru', 'rd', 'lu', 'ld', 'tcw', 'tccw', 'bcw', 'bccw']

        self.children = {}
        self.parent = None
        self.move_from_parent = None
        self.depth = 0

    ## Making a move using the rubik's cube
    def move(self, move: str) -> bool:
        # Top clockwise (TCW)
        if move == 'tcw':
            # Rotate top layer
            initial_front_top = self.faces['front'][0:2]
            initial_right_top = self.faces['right'][0:2]
            initial_back_top  = self.faces['back'][0:2]
            initial_left_top  = self.faces['left'][0:2]

            self.faces['front'][0:2] = initial_right_top
            self.faces['right'][0:2] = initial_back_top
            self.faces['back'][0:2]  = initial_left_top
            self.faces['left'][0:2]  = initial_front_top

            # Rotate top face
            t = self.faces['top']  
            self.faces['top'] = [t[2], t[0], t[3], t[1]]
            return True

        # Top counterclockwise (TCCW)
        elif move == 'tccw':
            # Rotate top layer
            initial_front_top = self.faces['front'][0:2]
            initial_right_top = self.faces['right'][0:2]
            initial_back_top  = self.faces['back'][0:2]
            initial_left_top  = self.faces['left'][0:2]

            self.faces['front'][0:2] = initial_left_top
            self.faces['right'][0:2] = initial_front_top
            self.faces['back'][0:2]  = initial_right_top
            self.faces['left'][0:2]  = initial_back_top

            # Rotate top face
            t = self.faces['top']  
            self.faces['top'] = [t[1], t[3], t[0], t[2]]
            return True

        # Bottom clockwise (BCW)
        elif move == 'bcw':
            # Rotate bottom layer
            initial_front_bottom = self.faces['front'][2:4]
            initial_right_bottom = self.faces['right'][2:4]
            initial_back_bottom  = self.faces['back'][2:4]
            initial_left_bottom  = self.faces['left'][2:4]

            self.faces['front'][2:4] = initial_right_bottom
            self.faces['right'][2:4] = initial_back_bottom
            self.faces['back'][2:4]  = initial_left_bottom
            self.faces['left'][2:4]  = initial_front_bottom

            # Rotate bottom face
            b = self.faces['bottom'] 
            self.faces['bottom'] = [b[2], b[0], b[3], b[1]]
            return True

        # Bottom counterclockwise (BCCW)
        elif move == 'bccw':
            # Rotate bottom layer
            initial_front_bottom = self.faces['front'][2:4]
            initial_right_bottom = self.faces['right'][2:4]
            initial_back_bottom  = self.faces['back'][2:4]
            initial_left_bottom  = self.faces['left'][2:4]

            self.faces['front'][2:4] = initial_left_bottom
            self.faces['right'][2:4] = initial_front_bottom
            self.faces['back'][2:4]  = initial_right_bottom
            self.faces['left'][2:4]  = initial_back_bottom

            # Rotate bottom face
            b = self.faces['bottom']  
            self.faces['bottom'] = [b[1], b[3], b[0], b[2]]
            return True

        # Right up (RU)
        elif move == 'ru':
            # Rotate right layer up
            initial_front_right=[self.faces['front'][1],self.faces['front'][3]]
            initial_top_right=[self.faces['top'][1],self.faces['top'][3]]
            initial_back_left=[self.faces['back'][0],self.faces['back'][2]]
            initial_bottom_right=[self.faces['bottom'][1],self.faces['bottom'][3]]

            self.faces['top'][1],self.faces['top'][3]=initial_front_right
            self.faces['front'][1],self.faces['front'][3]=initial_bottom_right
            self.faces['bottom'][1],self.faces['bottom'][3]=initial_back_left[::-1]
            self.faces['back'][0],self.faces['back'][2]=initial_top_right[::-1]

            # Rotate right face (CW)
            r=self.faces['right']
            self.faces['right']=[r[2],r[0],r[3],r[1]]
            return True

        # Right down (RD)
        elif move == 'rd':
            # Rotate right layer down
            initial_front_right=[self.faces['front'][1],self.faces['front'][3]]
            initial_top_right=[self.faces['top'][1],self.faces['top'][3]]
            initial_back_left=[self.faces['back'][0],self.faces['back'][2]]
            initial_bottom_right=[self.faces['bottom'][1],self.faces['bottom'][3]]

            self.faces['top'][1],self.faces['top'][3]=initial_back_left[::-1]
            self.faces['front'][1],self.faces['front'][3]=initial_top_right
            self.faces['bottom'][1],self.faces['bottom'][3]=initial_front_right
            self.faces['back'][0],self.faces['back'][2]=initial_bottom_right[::-1]
            
            # Rotate right face (CCW)
            r=self.faces['right']
            self.faces['right']=[r[1],r[3],r[0],r[2]]
            return True

        # Left up (LU)
        elif move == 'lu':
            # Rotate left layer up
            initial_front_left = [self.faces['front'][0],  self.faces['front'][2]]
            initial_top_left = [self.faces['top'][0],    self.faces['top'][2]]
            initial_back_right = [self.faces['back'][1],   self.faces['back'][3]]   
            initial_bottom_left = [self.faces['bottom'][0], self.faces['bottom'][2]]

            self.faces['top'][0], self.faces['top'][2] = initial_front_left
            self.faces['front'][0], self.faces['front'][2] = initial_bottom_left
            self.faces['bottom'][0], self.faces['bottom'][2] = initial_back_right[::-1]
            self.faces['back'][1], self.faces['back'][3] = initial_top_left[::-1]

            # Rotate left face (CCW)
            l = self.faces['left']
            self.faces['left'] = [l[1], l[3], l[0], l[2]]
            return True

        # Left down (LD)
        elif move == 'ld':
            # Rotate left layer down
            initial_front_left = [self.faces['front'][0],  self.faces['front'][2]]
            initial_top_left = [self.faces['top'][0],    self.faces['top'][2]]
            initial_back_right = [self.faces['back'][1],   self.faces['back'][3]]   
            initial_bottom_left = [self.faces['bottom'][0], self.faces['bottom'][2]]

            self.faces['top'][0], self.faces['top'][2] = initial_back_right[::-1]
            self.faces['front'][0], self.faces['front'][2] = initial_top_left
            self.faces['bottom'][0], self.faces['bottom'][2] = initial_front_left
            self.faces['back'][1], self.faces['back'][3] = initial_bottom_left[::-1]

            # Rotate left face (CW)
            l = self.faces['left']
            self.faces['left'] = [l[2], l[0], l[3], l[1]]
            return True

        return False

    ## Randomize 
    def randomize(self, n: int):

        for _ in range(random.randint(0, n)):
            move = random.randint(0, 7)
            self.move(self.move_options[move])

        return self


    ## Solved?
    def solved(self) -> bool:
        solved_rubriks = Rubiks()
        
        if self.faces == solved_rubriks.faces:
            return True
        
        return False
    
    ## Clones a cube
    def clone(self):
        new_cube = Rubiks()
        # manually copy faces
        new_cube.faces = {face: stickers[:] for face, stickers in self.faces.items()}
        new_cube.parent = self
        return new_cube

    ## Creates children at current cube state
    def generate_children(self):
        self.children = {}
        for move in self.move_options:
            child = self.clone()
            child.move(move)
            child.parent = self
            child.move_from_parent = move
            child.depth = self.depth + 1
            self.children[move] = child

    ## Encode cube for set
    def __eq__(self, other):
        if not isinstance(other, Rubiks):
            return False
        return self.faces == other.faces

    def __hash__(self):
        return hash(tuple(sorted((face, tuple(stickers)) for face, stickers in self.faces.items())))
    

    ## SOLVING SECTION ##
    ## BFS
    def solve_BFS(self):
        frontier = deque()
        explored = set()

        frontier.append(self)

        while len(frontier) != 0:
            current_node = frontier.popleft()
            print(f"Depth: {current_node.depth}, Frontier: {len(frontier)}, Explored: {len(explored)}")

            if current_node.solved():
                # Backtrack to get move sequence
                moves = []
                node = current_node
                while node.parent is not None:
                    moves.append(node.move_from_parent)
                    node = node.parent
                moves.reverse()
                return len(explored), moves

            explored.add(current_node)

            current_node.generate_children()
            for child in current_node.children.values():
                if child not in frontier and child not in explored:
                    frontier.append(child)

        return -1, []

                
            



    
    







            


        