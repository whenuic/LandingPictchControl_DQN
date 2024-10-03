import pymunk
import pygame
import pymunk.pygame_util
import math

pygame.init()

WIDTH = 2000
HEIGHT = 800
window = pygame.display.set_mode((WIDTH, HEIGHT))


def create_box(space, mass):
    body = pymunk.Body()
    body.position = (100, 100)
    body.velocity = (30, 0)
    shape = pymunk.Poly.create_box(body, (12, 4))
    shape.mass = mass
    shape.color = (255, 255, 0, 100)
    space.add(body, shape)
    return shape


def create_ground(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (WIDTH / 2, 201)
    shape = pymunk.Poly.create_box(body, (WIDTH, 2))
    shape.color = (255, 255, 255, 100)
    space.add(body, shape)
    return shape


def draw(space, window, draw_options):
    window.fill("black")
    space.debug_draw(draw_options)
    pygame.display.update()


def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps

    space = pymunk.Space()
    space.gravity = (0, 9.81)

    box = create_box(space, 600)
    ground = create_ground(space)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        draw(space, window, draw_options)
        # box.body.apply_force_at_local_point((0, -600), (6, 0))
        space.step(dt)
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)
