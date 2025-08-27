import cv2
import numpy as np
from config import Config
from kernel import kernel


def update(frame: np.ndarray, kernel: np.ndarray):
    # Apply the kernel using OpenCV
    sum_of_weighted_neighbours = cv2.filter2D(frame, -1, kernel)
    
    diff = (sum_of_weighted_neighbours - Config.mu) / Config.sigma
    growth = 2 * np.exp(-(diff ** 2) / 2) - 1
    new_frame = frame + growth * (1 / Config.update_frequency)
    
    return np.clip(new_frame, 0, 1)

# Create a sample image (or use your data)
frame = np.random.random(Config.grid_size)
frame = frame.astype(np.float32)
while True:
    # Updates current state
    frame = update(frame, kernel)

    # Display the window
    cv2.imshow('Continuous Automata', frame)

    # Wait for any key press and stops if the key pressed is 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()