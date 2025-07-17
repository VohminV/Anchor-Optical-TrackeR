import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Настройки окна
WIDTH, HEIGHT = 800, 600

camera_height = 2.0
camera_distance = 5.0
current_offset = {'dx': 0, 'dy': 0, 'angle': 0}

def read_offsets():
    try:
        with open('offsets.json', 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Error reading offsets.json: {e}")
        return []

def update_offsets():
    global current_offset
    offsets = read_offsets()
    if offsets:
        current_offset = offsets[-1]

def setup_3d(width, height):
    glViewport(0, 0, width, height)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, width / float(height), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def draw_axes():
    glBegin(GL_LINES)
    # X axis - red
    glColor3f(1, 0, 0)
    glVertex3f(-0.5, 0, 0)
    glVertex3f(0.5, 0, 0)
    # Z axis - green
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, -0.5)
    glVertex3f(0, 0, 0.5)
    # Y axis - blue
    glColor3f(0, 0, 1)
    glVertex3f(0, -0.5, 0)
    glVertex3f(0, 0.5, 0)
    glEnd()

def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("3D Drone Simulator")

    setup_3d(WIDTH, HEIGHT)

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                w, h = event.size
                pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL | RESIZABLE)
                setup_3d(w, h)

        update_offsets()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        dx = current_offset.get('dx', 0)
        dy = current_offset.get('dy', 0)
        angle = current_offset.get('angle', 0)

        cam_x = dx
        cam_y = camera_height
        cam_z = dy + camera_distance

        gluLookAt(cam_x, cam_y, cam_z,
                  dx, 0.0, dy,
                  0, 1, 0)

        glPushMatrix()
        glTranslatef(dx, 0, dy)
        glRotatef(angle * 57.2958, 0, 1, 0)

        draw_axes()

        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
