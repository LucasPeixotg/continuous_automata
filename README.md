# YAML Configuration Guide for Continuous Cellular Automata

This document explains how to configure continuous cellular automata simulations using YAML files.

## Table of Contents
- [Basic Structure](#basic-structure)
- [Grid Configuration](#grid-configuration)
- [Functions](#functions)
- [Kernels](#kernels)
- [Lifeforms](#lifeforms)
- [Simulation Settings](#simulation-settings)
- [Display Settings](#display-settings)
- [Complete Examples](#complete-examples)

---

## Basic Structure

A YAML configuration file has the following top-level sections:

```yaml
grid:           # Required: Grid dimensions
functions:      # Optional: Reusable growth functions
kernels:        # Optional: Reusable convolution kernels
lifeforms:      # Required: Define organisms and their rules
simulation:     # Optional: Simulation parameters
display:        # Optional: Display settings
```

---

## Grid Configuration

Defines the simulation grid size.

```yaml
grid:
  width: 1080      # Grid width in pixels
  height: 720      # Grid height in pixels
```

**Parameters:**
- `width`: Integer, grid width
- `height`: Integer, grid height

---

## Functions

Growth functions determine how cells evolve based on their neighborhood values. You can define reusable functions that can be referenced by name.

### Function Types

#### 1. Gaussian Function
Creates smooth, bell-curve growth patterns (used in Lenia-style automata).

```yaml
functions:
  smooth_growth:
    type: gaussian
    mu: 0.11           # Center of the bell curve
    sigma: 0.08        # Width of the bell curve
    amplitude: 2.0     # Height of the peak
    baseline: -1.0     # Baseline value
```

**Formula:** `amplitude * exp(-((x - mu) / sigma)² / 2) + baseline`

**Use case:** Smooth, continuous organisms that form stable patterns.

#### 2. Step Function
Binary threshold function.

```yaml
functions:
  threshold_growth:
    type: step
    threshold: 0.5     # Activation threshold
    low_value: -1.0    # Value when below threshold
    high_value: 1.0    # Value when above threshold
```

**Use case:** Binary decisions, simple on/off behaviors.

#### 3. Linear Function
Simple linear transformation.

```yaml
functions:
  linear_growth:
    type: linear
    slope: 2.0         # Multiplication factor
    intercept: -1.0    # Added constant
```

**Formula:** `slope * x + intercept`

**Use case:** Proportional responses, simple amplification.

#### 4. Conway Function
Classic Conway's Game of Life rules.

```yaml
functions:
  game_of_life:
    type: conway
    underpopulation: 2   # Dies if neighbors < this
    overpopulation: 3    # Dies if neighbors > this
    reproduction: 3      # Born if neighbors == this
```

**Use case:** Discrete cellular automata, Game of Life variants.

#### 5. Identity Function
Returns input unchanged.

```yaml
functions:
  passthrough:
    type: identity
```

**Use case:** Testing, or when you want the raw convolution result.

### Using Function References

Define once, use multiple times:

```yaml
functions:
  my_func:
    type: gaussian
    mu: 0.15
    sigma: 0.05

lifeforms:
  organism:
    rules:
      self:
        func:
          ref: my_func  # Reference the predefined function
```

Or define inline:

```yaml
lifeforms:
  organism:
    rules:
      self:
        func:
          type: gaussian  # Define function inline
          mu: 0.15
          sigma: 0.05
```

---

## Kernels

Kernels define the neighborhood structure for convolution operations. They determine which nearby cells influence each other.

### Kernel Types

#### 1. Disk Kernel
Circular neighborhood.

```yaml
kernels:
  small_circle:
    type: disk
    radius: 10                    # Radius in pixels
    gaussian_sigma: 0.8           # Blur amount (0 = no blur)
    gaussian_kernel_size: 5       # Blur kernel size (odd number)
```

**Use case:** Radial influence, smooth circular patterns.

#### 2. Ring Kernel
Donut-shaped neighborhood (outer circle minus inner circle).

```yaml
kernels:
  donut:
    type: ring
    outer_diameter: 21            # Outer circle diameter
    inner_diameter: 15            # Inner circle diameter
    gaussian_sigma: 1.2           # Blur amount
    gaussian_kernel_size: 15      # Blur kernel size
```

**Use case:** Lenia-style organisms, traveling waves, detecting edges at specific distances.

#### 3. Square Kernel
Rectangular neighborhood.

```yaml
kernels:
  box:
    type: square
    side_length: 3                # Size of square
    gaussian_sigma: 0.0           # Blur amount
    gaussian_kernel_size: 1       # Blur kernel size
```

**Custom array** for specific patterns (like Conway's Game of Life):

```yaml
kernels:
  conway_neighbors:
    type: square
    side_length: 3
    gaussian_sigma: 0.0
    gaussian_kernel_size: 1
    custom_array: [[1, 1, 1], 
                   [1, 0, 1], 
                   [1, 1, 1]]  # Counts 8 neighbors, excludes center
```

**Use case:** Classic cellular automata, grid-based patterns.

#### 4. Composite Kernels
Combine kernels using mathematical operations.

**Addition:**
```yaml
kernels:
  combined:
    type: composite
    operation: add
    left:
      type: disk
      radius: 5
    right:
      type: ring
      outer_diameter: 20
      inner_diameter: 15
```

**Subtraction:**
```yaml
kernels:
  difference:
    type: composite
    operation: subtract
    left:
      ref: large_ring       # Reference to predefined kernel
    right:
      ref: small_disk       # Reference to predefined kernel
```

**Multiplication by scalar:**
```yaml
kernels:
  scaled:
    type: composite
    operation: multiply
    left:
      type: disk
      radius: 10
    right: 0.5              # Number, not a kernel
```

**Division by scalar:**
```yaml
kernels:
  divided:
    type: composite
    operation: divide
    left:
      type: ring
      outer_diameter: 30
      inner_diameter: 20
    right: 2                # Number, not a kernel
```

**Complex example:**
```yaml
kernels:
  lenia_kernel:
    type: composite
    operation: subtract
    left:
      type: ring
      outer_diameter: 21
      inner_diameter: 15
      gaussian_sigma: 1.2
      gaussian_kernel_size: 15
    right:
      type: composite
      operation: divide
      left:
        type: ring
        outer_diameter: 41
        inner_diameter: 31
        gaussian_sigma: 1.1
        gaussian_kernel_size: 13
      right: 2
```

### Gaussian Blur Parameters

- `gaussian_sigma`: Controls blur strength
  - `0.0` = no blur (sharp edges)
  - `0.5` = slight blur
  - `1.0+` = significant blur (smooth gradients)
- `gaussian_kernel_size`: Size of blur kernel (must be odd)
  - Larger values = more computation but smoother blur
  - Typical values: 3, 5, 7, 9, 11

---

## Lifeforms

Lifeforms are the organisms or entities in your simulation. Each lifeform has its own state grid and rules.

### Basic Structure

```yaml
lifeforms:
  organism_name:
    color: "#00FF00"              # Hex color for display
    initial_state: random         # How to initialize
    rules:                        # Interaction rules
      target_organism:            # Which organism this rule affects
        dt: 0.05                  # Time step size
        kernel: ...               # Convolution kernel
        func: ...                 # Growth function
        weight: 1.0               # Rule weight/strength
```

### Initial State Types

```yaml
initial_state: random           # Random continuous values [0, 1]
initial_state: random_discrete  # Random binary values (0 or 1)
initial_state: empty            # All zeros
initial_state: centered_blob    # Single blob in the center
initial_state: random_sparse    # Sparse random (5% filled)
```

### Rules

Each lifeform can have multiple rules that define how it interacts with itself and other lifeforms.

#### Self-Interaction Rule

```yaml
lifeforms:
  wanderer:
    rules:
      wanderer:              # Same name = self-interaction
        dt: 0.05
        kernel:
          type: ring
          outer_diameter: 20
          inner_diameter: 15
        func:
          type: gaussian
          mu: 0.11
          sigma: 0.08
        weight: 1.0
```

#### Cross-Species Interaction

```yaml
lifeforms:
  predator:
    rules:
      predator:              # Self-interaction
        dt: 0.03
        kernel: ...
        func: ...
        weight: 1.0
      
      prey:                  # Interaction with prey
        dt: 0.02
        kernel: ...
        func: ...
        weight: 0.5          # Weaker influence
```

#### Negative Interaction (Repulsion)

```yaml
lifeforms:
  prey:
    rules:
      prey:                  # Self-interaction
        dt: 0.04
        kernel: ...
        func: ...
        weight: 1.0
      
      predator:              # Avoid predators
        dt: 0.02
        kernel: ...
        func:
          type: linear
          slope: -2.0        # Negative slope
          intercept: 0.0
        weight: -0.8         # Negative weight = repulsion
```

### Time Step (dt)

The `dt` parameter controls how much the rule affects the state per frame.

- **Small dt (0.01 - 0.05)**: Slow, smooth changes (continuous dynamics)
- **Medium dt (0.1 - 0.5)**: Moderate speed
- **Large dt (1.0)**: Fast, discrete changes (like Conway's Game of Life)

### Weight

The `weight` parameter scales the rule's influence.

- `weight: 1.0`: Normal strength
- `weight: 0.5`: Half strength
- `weight: 2.0`: Double strength
- `weight: -1.0`: Inverted (repulsion instead of attraction)

---

## Simulation Settings

Optional settings for controlling the simulation.

```yaml
simulation:
  max_iterations: null         # null = infinite, or a number
  save_every: 100              # Save frame every N iterations
  output_directory: "./output" # Where to save frames
  record_video: false          # Record video
  video_fps: 30                # Video frame rate
```

---

## Display Settings

Optional settings for the display window.

```yaml
display:
  window_name: "My Simulation"  # Window title
  show_grid: false              # Show grid lines (not implemented)
  show_fps: true                # Show FPS counter
  quit_key: "q"                 # Key to quit simulation
  scale: 1.0                    # Display scale factor (1.0 = original size, 2.0 = 2x larger)
```

**Scale Parameter:**
- `scale: 0.5` - Display at half size (faster rendering, harder to see)
- `scale: 1.0` - Original size (default)
- `scale: 2.0` - Display at 2x size (easier to see, uses more screen space)
- `scale: 3.0` - Display at 3x size (very large)

Note: The scale only affects display, not the actual simulation grid size. Use `INTER_NEAREST` interpolation to maintain crisp pixel edges.

---

## Complete Examples

### Example 1: Conway's Game of Life

```yaml
grid:
  width: 1080
  height: 720

lifeforms:
  conway:
    color: "#FFFFFF"
    initial_state: random_discrete
    rules:
      conway:
        dt: 1.0
        kernel:
          type: square
          side_length: 3
          gaussian_sigma: 0.0
          gaussian_kernel_size: 1
          custom_array: [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        func:
          type: conway
          underpopulation: 2
          overpopulation: 3
          reproduction: 3
        weight: 1.0

display:
  window_name: "Conway's Game of Life"
  quit_key: "q"
  show_fps: true
```

### Example 2: Lenia-Style Wandering Blob

```yaml
grid:
  width: 1080
  height: 720

functions:
  lenia_growth:
    type: gaussian
    mu: 0.11
    sigma: 0.08
    amplitude: 2.0
    baseline: -1.0

kernels:
  inner_ring:
    type: ring
    outer_diameter: 21
    inner_diameter: 15
    gaussian_sigma: 1.2
    gaussian_kernel_size: 15
  
  outer_ring:
    type: ring
    outer_diameter: 41
    inner_diameter: 31
    gaussian_sigma: 1.1
    gaussian_kernel_size: 13

lifeforms:
  wanderer:
    color: "#00FF88"
    initial_state: random
    rules:
      wanderer:
        dt: 0.05
        kernel:
          type: composite
          operation: subtract
          left:
            ref: inner_ring
          right:
            type: composite
            operation: divide
            left:
              ref: outer_ring
            right: 2
        func:
          ref: lenia_growth
        weight: 1.0

display:
  window_name: "Lenia Wanderer"
  show_fps: true
```

### Example 3: Predator-Prey System

```yaml
grid:
  width: 1080
  height: 720

functions:
  smooth_growth:
    type: gaussian
    mu: 0.15
    sigma: 0.05
    amplitude: 1.5
    baseline: -0.5

lifeforms:
  prey:
    color: "#0088FF"  # Blue
    initial_state: random
    rules:
      prey:
        dt: 0.04
        kernel:
          type: ring
          outer_diameter: 21
          inner_diameter: 15
          gaussian_sigma: 1.0
          gaussian_kernel_size: 11
        func:
          ref: smooth_growth
        weight: 1.0
      
      predator:  # Prey is consumed by predators
        dt: 0.02
        kernel:
          type: disk
          radius: 15
          gaussian_sigma: 1.5
          gaussian_kernel_size: 9
        func:
          type: linear
          slope: -2.0
          intercept: 0.0
        weight: -1.0
  
  predator:
    color: "#FF0000"  # Red
    initial_state: random
    rules:
      predator:
        dt: 0.03
        kernel:
          type: disk
          radius: 10
          gaussian_sigma: 0.8
          gaussian_kernel_size: 5
        func:
          ref: smooth_growth
        weight: 1.0
      
      prey:  # Predators grow near prey
        dt: 0.02
        kernel:
          type: ring
          outer_diameter: 30
          inner_diameter: 20
          gaussian_sigma: 1.0
          gaussian_kernel_size: 11
        func:
          type: linear
          slope: 3.0
          intercept: -0.5
        weight: 0.5

display:
  window_name: "Predator-Prey Dynamics"
  show_fps: true
```

---

## Tips and Best Practices

### 1. Start Simple
Begin with a single lifeform and one rule. Add complexity gradually.

### 2. Tune Parameters Iteratively
- Start with known working parameters (like Conway's or the Lenia examples)
- Change one parameter at a time
- Use small dt values for smoother dynamics

### 3. Kernel Design
- **Disk**: Good for spreading, diffusion-like behaviors
- **Ring**: Good for traveling waves, self-organizing patterns
- **Small kernels**: Fast computation, local interactions
- **Large kernels**: Slow computation, long-range interactions

### 4. Function Design
- **Gaussian**: Most versatile for continuous dynamics
  - `mu` around 0.1-0.15 often works well
  - `sigma` around 0.05-0.1 for smooth transitions
- **Step**: Good for binary behaviors, but can be unstable
- **Linear**: Simple but powerful for proportional responses

### 5. Time Step (dt)
- Too large: Unstable, explodes
- Too small: Slow evolution, nothing happens
- Start with 0.05 for continuous systems
- Use 1.0 for discrete systems (Conway-like)

### 6. Performance
- Larger grids = slower computation
- Larger kernels = slower computation
- More lifeforms = slower computation
- Start with 720x1080 or smaller for testing

### 7. Color Selection
Use distinct colors for different lifeforms:
- `"#FF0000"` - Red
- `"#00FF00"` - Green
- `"#0000FF"` - Blue
- `"#FFFF00"` - Yellow
- `"#FF00FF"` - Magenta
- `"#00FFFF"` - Cyan
- `"#FFFFFF"` - White

### 8. Common Patterns

**Self-organizing blobs (Lenia-style):**
- Ring kernels (inner + outer)
- Gaussian growth function
- dt = 0.03-0.05
- mu = 0.10-0.15, sigma = 0.05-0.10

**Discrete automata (Conway-style):**
- Square kernel with custom array
- Conway growth function
- dt = 1.0
- Binary initial state

**Diffusion/spreading:**
- Disk kernel
- Linear or step function
- Small dt = 0.01-0.03

**Waves:**
- Large ring kernel
- Gaussian function with low mu
- Medium dt = 0.05-0.10

---

## Running Your Configuration

```bash
# Use default config (automata_config.yml)
python yaml_main.py

# Use specific config file
python yaml_main.py my_config.yml

# Press 'q' (or configured quit_key) to exit
```

---

## Troubleshooting

### Simulation explodes (values go to infinity)
- **Solution**: Reduce `dt` values, ensure growth function returns values in reasonable range

### Nothing happens
- **Solution**: Increase `dt`, check that initial_state isn't 'empty', verify growth function isn't always returning 0

### Patterns die out immediately
- **Solution**: Adjust growth function parameters (mu, sigma), try different kernel sizes

### Performance is slow
- **Solution**: Reduce grid size, use smaller kernels, reduce number of lifeforms

### Kernel reference not found
- **Solution**: Ensure kernel is defined in `kernels:` section before referencing it

### Function reference not found
- **Solution**: Ensure function is defined in `functions:` section before referencing it

---

## Mathematical Details

### How It Works

1. **Convolution**: Each kernel is convolved with a lifeform's state grid
   ```
   neighborhood_value = kernel ⊗ state
   ```

2. **Growth Function**: Applied to convolution result
   ```
   growth = func(neighborhood_value)
   ```

3. **State Update**: New state is computed
   ```
   new_state = clip(state + growth * dt * weight, 0, 1)
   ```

4. **Multi-Rule Update**: Multiple rules are summed
   ```
   new_state = state + Σ(growth_i * dt_i * weight_i)
   ```

### Normalization

Some kernels are automatically normalized so their positive values sum to 1. This ensures consistent behavior across different kernel sizes.

---

## Further Reading

- **Lenia**: [Paper by Bert Wang-Chak Chan](https://arxiv.org/abs/1812.05433)
- **Conway's Game of Life**: [Wikipedia](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- **Cellular Automata**: [Wolfram MathWorld](https://mathworld.wolfram.com/CellularAutomaton.html)

---

*Happy experimenting! The beauty of these systems is in discovering emergent behaviors through parameter exploration.*