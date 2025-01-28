import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

def initialize_grid(grid_size, randomize=True):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    if randomize:
        return np.random.choice([0, 1], size=(grid_size, grid_size))
    else:
        gamePattern = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        pattern_rows, pattern_cols = len(gamePattern), len(gamePattern[0])
        start_row = (grid_size - pattern_rows) // 2
        start_col = (grid_size - pattern_cols) // 2

        for r in range(pattern_rows):
            for c in range(pattern_cols):
                grid[start_row + r, start_col + c] = gamePattern[r][c]
    return grid

def update_grid(grid):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    neighbors = convolve2d(grid, kernel, mode='same', boundary='wrap')
    
    birth = (grid == 0) & (neighbors == 3) 
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3)) 
    new_grid = birth | survive 

    return new_grid.astype(int)

def animate_game_of_life(grid_size, generations, interval=200, randomize=True):
    grid = initialize_grid(grid_size, randomize=randomize)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')  
    ax.set_facecolor('black')  
    img = ax.imshow(grid, cmap='binary_r', interpolation='nearest') 

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    def update(frame):
        nonlocal grid
        grid = update_grid(grid)
        img.set_data(grid)
        return img,

    ani = animation.FuncAnimation(fig, update, frames=generations, interval=interval, blit=True)
    plt.show()

GRID_SIZE = 100  
GENERATIONS = 500  
INTERVAL = 50  
RANDOMIZE = True  

animate_game_of_life(GRID_SIZE, GENERATIONS, INTERVAL, randomize=RANDOMIZE)
