class Rubiks:

    ## Initializes a solved rubik's cube
    def __init__(self): 
        self.faces = {
            'front': ['G'] * 9, # Green
            'back': ['B'] * 9, # Blue
            'bottom': ['R'] * 9, # Red
            'right': ['W'] * 9, # White
            'left': ['Y'] * 9, # Yellow
            'top': ['O'] * 9  # Orange
        } 

    ## Making a move using the rubik's cube
    def move(self, move: str) -> bool:
        # Twist top to left
        if move == 'tl': 
            # Getting initial values
            initial_front_top = self.faces['front'][:3]
            initial_back_top = self.faces['back'][:3]
            initial_left_top = self.faces['left'][:3]
            initial_right_top = self.faces['right'][:3]
            initial_top_top = self.faces['top'][:3]

            # Moving around
            self.faces['front'][:3] = initial_right_top
            self.faces['back'][:3] = initial_left_top
            self.faces['left'][:3] = initial_front_top
            self.faces['right'][:3] = initial_back_top
            
            # Rotate the top face counter clockwise
            initial_top_top = self.faces['top'][:]
            self.faces['top'][0] = initial_top_top[2]
            self.faces['top'][1] = initial_top_top[5]
            self.faces['top'][2] = initial_top_top[8]
            self.faces['top'][3] = initial_top_top[1]
            self.faces['top'][4] = initial_top_top[4]
            self.faces['top'][5] = initial_top_top[7]
            self.faces['top'][6] = initial_top_top[0]
            self.faces['top'][7] = initial_top_top[3]
            self.faces['top'][8] = initial_top_top[6]

        # Twist top to right
        if move == 'tr':
            # Getting initial values
            initial_front_top = self.faces['front'][:3]
            initial_back_top = self.faces['back'][:3]
            initial_left_top = self.faces['left'][:3]
            initial_right_top = self.faces['right'][:3]
            initial_top_top = self.faces['top'][:3]

            # Moving around
            self.faces['front'][:3] = initial_left_top
            self.faces['back'][:3] = initial_right_top
            self.faces['left'][:3] = initial_back_top
            self.faces['right'][:3] = initial_front_top
            
            # Rotate the top face clockwise
            initial_top_top = self.faces['top'][:]
            self.faces['top'][0] = initial_top_top[6]
            self.faces['top'][1] = initial_top_top[3]
            self.faces['top'][2] = initial_top_top[0]
            self.faces['top'][3] = initial_top_top[7]
            self.faces['top'][4] = initial_top_top[4]
            self.faces['top'][5] = initial_top_top[1]
            self.faces['top'][6] = initial_top_top[8]
            self.faces['top'][7] = initial_top_top[5]
            self.faces['top'][8] = initial_top_top[2]

        # Twist bottom to right
        if move == 'br':
            # Getting initial values
            initial_front_bottom = self.faces['front'][6:9]
            initial_back_bottom = self.faces['back'][6:9]
            initial_left_bottom = self.faces['left'][6:9]
            initial_right_bottom = self.faces['right'][6:9]
            initial_bottom_bottom = self.faces['bottom'][6:9]

            # Moving around
            self.faces['front'][6:9] = initial_left_bottom
            self.faces['back'][6:9] = initial_right_bottom
            self.faces['left'][6:9] = initial_back_bottom
            self.faces['right'][6:9] = initial_front_bottom
            
            # Rotate the bottom face clockwise
            initial_bottom_bottom = self.faces['bottom'][:]
            self.faces['bottom'][0] = initial_bottom_bottom[6]
            self.faces['bottom'][1] = initial_bottom_bottom[3]
            self.faces['bottom'][2] = initial_bottom_bottom[0]
            self.faces['bottom'][3] = initial_bottom_bottom[7]
            self.faces['bottom'][4] = initial_bottom_bottom[4]
            self.faces['bottom'][5] = initial_bottom_bottom[1]
            self.faces['bottom'][6] = initial_bottom_bottom[8]
            self.faces['bottom'][7] = initial_bottom_bottom[5]
            self.faces['bottom'][8] = initial_bottom_bottom[2]

        # Twist bottom to left
        if move == 'bl':
            # Getting initial values
            initial_front_bottom = self.faces['front'][6:9]
            initial_back_bottom = self.faces['back'][6:9]
            initial_left_bottom = self.faces['left'][6:9]
            initial_right_bottom = self.faces['right'][6:9]
            initial_bottom_bottom = self.faces['bottom'][6:9]

            # Moving around
            self.faces['front'][6:9] = initial_right_bottom
            self.faces['back'][6:9] = initial_left_bottom
            self.faces['left'][6:9] = initial_front_bottom
            self.faces['right'][6:9] = initial_back_bottom
            
            # Rotate the bottom face counter clockwise
            initial_bottom_bottom = self.faces['bottom'][:]
            self.faces['bottom'][0] = initial_bottom_bottom[2]
            self.faces['bottom'][1] = initial_bottom_bottom[5]
            self.faces['bottom'][2] = initial_bottom_bottom[8]
            self.faces['bottom'][3] = initial_bottom_bottom[1]
            self.faces['bottom'][4] = initial_bottom_bottom[4]
            self.faces['bottom'][5] = initial_bottom_bottom[7]
            self.faces['bottom'][6] = initial_bottom_bottom[0]
            self.faces['bottom'][7] = initial_bottom_bottom[3]
            self.faces['bottom'][8] = initial_bottom_bottom[6]

        # Twist left upwards
        if move == 'lu':
            # Getting initial values
            initial_front_left = [self.faces['front'][0], self.faces['front'][3], self.faces['front'][6]]
            initial_back_left = [self.faces['back'][2], self.faces['back'][5], self.faces['back'][8]]
            initial_top_left = [self.faces['top'][0], self.faces['top'][3], self.faces['top'][6]]
            initial_bottom_left = [self.faces['bottom'][0], self.faces['bottom'][3], self.faces['bottom'][6]]

            # Moving around
            self.faces['front'][0], self.faces['front'][3], self.faces['front'][6] = initial_bottom_left
            self.faces['back'][2], self.faces['back'][5], self.faces['back'][8] = initial_top_left
            self.faces['top'][0], self.faces['top'][3], self.faces['top'][6] = initial_front_left
            self.faces['bottom'][0], self.faces['bottom'][3], self.faces['bottom'][6] = initial_back_left
            
            # Rotate the left face counter clockwise
            initial_left_face = self.faces['left'][:]
            self.faces['left'][0] = initial_left_face[2]
            self.faces['left'][1] = initial_left_face[5]
            self.faces['left'][2] = initial_left_face[8]
            self.faces['left'][3] = initial_left_face[1]
            self.faces['left'][4] = initial_left_face[4]
            self.faces['left'][5] = initial_left_face[7]
            self.faces['left'][6] = initial_left_face[0]
            self.faces['left'][7] = initial_left_face[3]
            self.faces['left'][8] = initial_left_face[6]

        # Twist left downwards
        if move == 'ld':
            # Getting initial values
            initial_front_left = [self.faces['front'][0], self.faces['front'][3], self.faces['front'][6]]
            initial_back_left = [self.faces['back'][2], self.faces['back'][5], self.faces['back'][8]]
            initial_top_left = [self.faces['top'][0], self.faces['top'][3], self.faces['top'][6]]
            initial_bottom_left = [self.faces['bottom'][0], self.faces['bottom'][3], self.faces['bottom'][6]]

            # Moving around
            self.faces['front'][0], self.faces['front'][3], self.faces['front'][6] = initial_top_left
            self.faces['back'][2], self.faces['back'][5], self.faces['back'][8] = initial_bottom_left
            self.faces['top'][0], self.faces['top'][3], self.faces['top'][6] = initial_back_left
            self.faces['bottom'][0], self.faces['bottom'][3], self.faces['bottom'][6] = initial_front_left
            
            # Rotate the left face clockwise
            initial_left_face = self.faces['left'][:]
            self.faces['left'][0] = initial_left_face[6]
            self.faces['left'][1] = initial_left_face[3]
            self.faces['left'][2] = initial_left_face[0]
            self.faces['left'][3] = initial_left_face[7]
            self.faces['left'][4] = initial_left_face[4]
            self.faces['left'][5] = initial_left_face[1]
            self.faces['left'][6] = initial_left_face[8]
            self.faces['left'][7] = initial_left_face[5]
            self.faces['left'][8] = initial_left_face[2]

        # Twist right upwards
        if move == 'ru':
            # Getting initial values
            initial_front_right = [self.faces['front'][2], self.faces['front'][5], self.faces['front'][8]]
            initial_back_right = [self.faces['back'][0], self.faces['back'][3], self.faces['back'][6]]
            initial_top_right = [self.faces['top'][2], self.faces['top'][5], self.faces['top'][8]]
            initial_bottom_right = [self.faces['bottom'][2], self.faces['bottom'][5], self.faces['bottom'][8]]

            # Moving around
            self.faces['front'][2], self.faces['front'][5], self.faces['front'][8] = initial_bottom_right
            self.faces['back'][0], self.faces['back'][3], self.faces['back'][6] = initial_top_right
            self.faces['top'][2], self.faces['top'][5], self.faces['top'][8] = initial_front_right
            self.faces['bottom'][2], self.faces['bottom'][5], self.faces['bottom'][8] = initial_back_right
            
            # Rotate the right face clockwise
            initial_right_face = self.faces['right'][:]
            self.faces['right'][0] = initial_right_face[6]
            self.faces['right'][1] = initial_right_face[3]
            self.faces['right'][2] = initial_right_face[0]
            self.faces['right'][3] = initial_right_face[7]
            self.faces['right'][4] = initial_right_face[4]
            self.faces['right'][5] = initial_right_face[1]
            self.faces['right'][6] = initial_right_face[8]
            self.faces['right'][7] = initial_right_face[5]
            self.faces['right'][8] = initial_right_face[2]

        # Twist right downwards
        if move == 'rd':
            # Getting initial values
            initial_front_right = [self.faces['front'][2], self.faces['front'][5], self.faces['front'][8]]
            initial_back_right = [self.faces['back'][0], self.faces['back'][3], self.faces['back'][6]]
            initial_top_right = [self.faces['top'][2], self.faces['top'][5], self.faces['top'][8]]
            initial_bottom_right = [self.faces['bottom'][2], self.faces['bottom'][5], self.faces['bottom'][8]]

            # Moving around
            self.faces['front'][2], self.faces['front'][5], self.faces['front'][8] = initial_top_right
            self.faces['back'][0], self.faces['back'][3], self.faces['back'][6] = initial_bottom_right
            self.faces['top'][2], self.faces['top'][5], self.faces['top'][8] = initial_back_right
            self.faces['bottom'][2], self.faces['bottom'][5], self.faces['bottom'][8] = initial_front_right
            
            # Rotate the right face counter clockwise
            initial_right_face = self.faces['right'][:]
            self.faces['right'][0] = initial_right_face[2]
            self.faces['right'][1] = initial_right_face[5]
            self.faces['right'][2] = initial_right_face[8]
            self.faces['right'][3] = initial_right_face[1]
            self.faces['right'][4] = initial_right_face[4]
            self.faces['right'][5] = initial_right_face[7]
            self.faces['right'][6] = initial_right_face[0]
            self.faces['right'][7] = initial_right_face[3]
            self.faces['right'][8] = initial_right_face[6]

    # # Randomize 
    # def randomize(cube: Rubiks) -> Rubiks:
        
