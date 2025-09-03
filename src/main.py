import cv2
from Grid import Grid
from Rule import Rule
from Kernel import Ring, Disk, KernelVisualizer
from LifeForm import LifeForm
import numpy as np

GRID_SIZE = (720, 1080)

def run(grid: Grid):
    while True:
        # Updates current state
        grid.update()

        # Display the window
        frame = grid.calculate_displayable_version()
        cv2.imshow('Continuous Automata', frame)

        # Wait for any key press and stops if the key pressed is 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()



def create_walking_blob(name: str) -> LifeForm:
    kernel = Ring(
        outer_diameter=21, 
        inner_diameter=15,     # 41 * 0.37
        gaussian_sigma=1.2,
        gaussian_kernel_size=15
    ) - Ring(
        outer_diameter=41, 
        inner_diameter=31,     # 41 * 0.37
        gaussian_sigma=1.1,
        gaussian_kernel_size=13
    ) / 2

    kernel.normalize()

    def func(x):
        diff = (x - 0.11) / 0.08
        return 2 * np.exp(-(diff**2) / 2) - 1

    rule = Rule(0.05, kernel, func)

    form = LifeForm(name, GRID_SIZE)
    form.add_rule(name, rule)
    return form

def create_walking_blob2(name: str) -> LifeForm:
    kernel = Ring(
        outer_diameter=21, 
        inner_diameter=15,     # 41 * 0.37
        gaussian_sigma=1.2,
        gaussian_kernel_size=15
    ) - Ring(
        outer_diameter=41, 
        inner_diameter=31,     # 41 * 0.37
        gaussian_sigma=1.1,
        gaussian_kernel_size=13
    ) / 4

    kernel.normalize()

    def func(x):
        diff = (x - 0.101) / 0.08
        return 2 * np.exp(-(diff**2) / 2) - 1

    rule = Rule(0.05, kernel, func)

    form = LifeForm(name, GRID_SIZE)
    form.add_rule(name, rule)
    return form

def create_blue_link(name: str, form1, form2) -> LifeForm:
    kernel = Ring(
        outer_diameter=21,
        inner_diameter=15,     # 41 * 0.37
        gaussian_sigma=1.2,
        gaussian_kernel_size=15
    ) - Ring(
        outer_diameter=41, 
        inner_diameter=31,     # 41 * 0.37
        gaussian_sigma=1.1,
        gaussian_kernel_size=13
    ) / 2

    kernel.normalize()

    def func(x):
        diff = (x - 0.01) / 0.09
        return 2 * np.exp(-(diff**2) / 2) - 1

    rule = Rule(0.20, kernel, func)
    form3 = LifeForm(name, GRID_SIZE)
    form3.add_rule(name, rule, 0)
    
    def func31(x):
        diff = (x - 0.09) / 0.08
        return 2 * np.exp(-(diff**2) / 2) - 1

    kernel31 = kernel

    rule = Rule(0.05, kernel31, func31)
    #form3.add_rule(form1.name, rule, -0.2)
    #form3.add_rule(form2.name, rule, 1)
    form3.add_rule(form1.name, rule)

    return form3

def create_blue_link(name: str, form1, form2) -> LifeForm:
    kernel = Ring(
        outer_diameter=21,
        inner_diameter=15,     # 41 * 0.37
        gaussian_sigma=1.2,
        gaussian_kernel_size=15
    ) - Ring(
        outer_diameter=41, 
        inner_diameter=31,     # 41 * 0.37
        gaussian_sigma=1.1,
        gaussian_kernel_size=13
    ) / 2

    kernel.normalize()

    def func(x):
        diff = (x - 0.01) / 0.09
        return 2 * np.exp(-(diff**2) / 2) - 1

    rule = Rule(0.05, kernel, func)
    form3 = LifeForm(name, GRID_SIZE)
    form3.add_rule(name, rule, 0)
    
    def func31(x):
        diff = (x - 0.09) / 0.08
        return 2 * np.exp(-(diff**2) / 2) - 1

    kernel31 = kernel

    rule = Rule(0.05, kernel31, func31)
    #form3.add_rule(form1.name, rule, -0.2)
    #form3.add_rule(form2.name, rule, 1)
    form3.add_rule(form1.name, rule)

    return form3

if __name__ == "__main__":
    form1 = create_walking_blob('form1')
    form2 = create_walking_blob2('form2')

    grid = Grid(GRID_SIZE)

    grid.add_lifeform(form1, "#39BFE8")
    #grid.add_lifeform(form2, "#e3c749")
    #grid.add_lifeform(form3, "#0000FF")
    grid.random()

    run(grid)