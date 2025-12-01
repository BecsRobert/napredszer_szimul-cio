import pygame
from pygame.locals import *
import OpenGL.GL as gl
from OpenGL.GLU import *
import numpy as np
import glm
import sys
import math
import random

# --- Segédfüggvények shaderek és textúrák betöltéséhez ---
def load_shader_source(filepath):
    try:
        with open(filepath, 'r') as f: return f.read()
    except FileNotFoundError:
        print(f"Hiba: Shader fájl nem található: {filepath}")
        raise

def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        raise RuntimeError(f"Shader fordítási hiba: {gl.glGetShaderInfoLog(shader).decode('utf-8')}")
    return shader

def create_shader_program(vertex_path, fragment_path):
    vert_shader = compile_shader(load_shader_source(vertex_path), gl.GL_VERTEX_SHADER)
    frag_shader = compile_shader(load_shader_source(fragment_path), gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vert_shader); gl.glAttachShader(program, frag_shader)
    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        raise RuntimeError(f"Program linkelési hiba: {gl.glGetProgramInfoLog(program).decode('utf-8')}")
    gl.glDeleteShader(vert_shader); gl.glDeleteShader(frag_shader)
    return program

def load_texture(filepath):
    try:
        surf = pygame.image.load(filepath).convert_alpha()
        tex_data = pygame.image.tostring(surf, "RGBA", False)
        w, h = surf.get_size()
        tid = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, tex_data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tid
    except Exception as e:
        print(f"Hiba a textúra betöltésekor '{filepath}': {e}")
        return 0

# --- Geometria Létrehozó Osztályok ---
class Sphere:
    def __init__(self, radius=1.0, sectors=36, stacks=18):
        vertices, indices = [], []
        sector_step, stack_step = 2 * math.pi / sectors, math.pi / stacks
        for i in range(stacks + 1):
            stack_angle = math.pi / 2 - i * stack_step
            xy = radius * math.cos(stack_angle); z = radius * math.sin(stack_angle)
            for j in range(sectors + 1):
                sector_angle = j * sector_step
                x = xy * math.cos(sector_angle); y = xy * math.sin(sector_angle)
                vertices.extend([x, y, z, x/radius, y/radius, z/radius, j / sectors, i / stacks])
        for i in range(stacks):
            for j in range(sectors):
                p1 = i * (sectors + 1) + j; p2 = p1 + sectors + 1
                indices.extend([p1, p2, p1 + 1]); indices.extend([p1 + 1, p2, p2 + 1])
        self.vertices = np.array(vertices, dtype=np.float32); self.indices = np.array(indices, dtype=np.uint32)
        self.vao = gl.glGenVertexArrays(1); gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        self.ebo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)
        stride = 8 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0)); gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(12)); gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(24)); gl.glEnableVertexAttribArray(2)
        gl.glBindVertexArray(0)
    def draw(self):
        gl.glBindVertexArray(self.vao); gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None); gl.glBindVertexArray(0)

class Ring:
    def __init__(self, inner_radius, outer_radius, sectors=256):
        vertices, indices = [], []
        sector_step = 2 * math.pi / sectors
        for i in range(sectors + 1):
            angle = i * sector_step
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            
            # JAVÍTÁS: Helyes textúrakoordináták a négyzetes gyűrű textúrához
            u_inner = (cos_a * inner_radius / (outer_radius * 2)) + 0.5
            v_inner = (sin_a * inner_radius / (outer_radius * 2)) + 0.5
            u_outer = (cos_a * outer_radius / (outer_radius * 2)) + 0.5
            v_outer = (sin_a * outer_radius / (outer_radius * 2)) + 0.5

            vertices.extend([cos_a * inner_radius, 0.0, sin_a * inner_radius, 0.0, 1.0, 0.0, u_inner, v_inner])
            vertices.extend([cos_a * outer_radius, 0.0, sin_a * outer_radius, 0.0, 1.0, 0.0, u_outer, v_outer])
        for i in range(sectors):
            p1 = i * 2; p2 = p1 + 1; p3 = p1 + 2; p4 = p1 + 3
            indices.extend([p1, p2, p3]); indices.extend([p2, p4, p3])
        self.vertices = np.array(vertices, dtype=np.float32); self.indices = np.array(indices, dtype=np.uint32)
        self.vao = gl.glGenVertexArrays(1); gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        self.ebo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)
        stride = 8 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0)); gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(12)); gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(24)); gl.glEnableVertexAttribArray(2)
        gl.glBindVertexArray(0)
    def draw(self):
        gl.glBindVertexArray(self.vao); gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None); gl.glBindVertexArray(0)

class StarField:
    def __init__(self, num_stars=8000, spread=1000.0):
        self.vertices = np.zeros(num_stars * 3, dtype=np.float32)
        for i in range(num_stars):
            x = random.uniform(-spread, spread); y = random.uniform(-spread, spread); z = random.uniform(-spread, spread)
            self.vertices[i*3 : i*3 + 3] = [x, y, z]
        self.num_stars = num_stars
        self.vao = gl.glGenVertexArrays(1); gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, gl.ctypes.c_void_p(0)); gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)
    def draw(self):
        gl.glPointSize(1.0); gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.num_stars); gl.glBindVertexArray(0)

class Orbit:
    def __init__(self, radius, segments=100):
        self.radius = radius; self.segments = segments; vertices = []
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            vertices.extend([math.cos(angle) * radius, 0, math.sin(angle) * radius])
        self.vertices = np.array(vertices, dtype=np.float32)
        self.vao = gl.glGenVertexArrays(1); gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None); gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)
    def draw(self):
        gl.glBindVertexArray(self.vao); gl.glDrawArrays(gl.GL_LINE_LOOP, 0, self.segments + 1); gl.glBindVertexArray(0)

# --- Bolygó osztály ---
class Planet:
    def __init__(self, name, radius, distance, orbit_speed, rotation_speed, texture_id, has_moon=False, has_rings=False, ring_texture_id=None):
        self.name = name; self.radius = radius; self.distance = distance
        self.orbit_speed = orbit_speed; self.rotation_speed = rotation_speed; self.texture_id = texture_id
        self.angle_orbit = random.uniform(0, 360); self.angle_rotation = 0.0
        self.has_moon = has_moon; self.moon_angle = random.uniform(0, 360)
        self.has_rings = has_rings; self.ring_texture_id = ring_texture_id
    def update(self, speed_factor):
        self.angle_orbit += self.orbit_speed * speed_factor; self.angle_rotation += self.rotation_speed * speed_factor
        if self.has_moon: self.moon_angle += self.orbit_speed * 2 * speed_factor
    def get_model_matrix(self, scale_override=1.0):
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(self.angle_orbit), glm.vec3(0, 1, 0))
        model = glm.translate(model, glm.vec3(self.distance, 0, 0))
        model = glm.rotate(model, glm.radians(self.angle_rotation), glm.vec3(0, 1, 0))
        model = glm.scale(model, glm.vec3(self.radius * scale_override))
        return model

class SunGlow:
    def __init__(self, size=8.0):
        # A méretet közvetlenül a vertexekre alkalmazzuk
        s = size
        self.vertices = np.array([
            -s,  s, 0.0, 0.0, 1.0,
            -s, -s, 0.0, 0.0, 0.0,
            s, -s, 0.0, 1.0, 0.0,
            -s,  s, 0.0, 0.0, 1.0,
            s, -s, 0.0, 1.0, 0.0,
            s,  s, 0.0, 1.0, 1.0
        ], dtype=np.float32)
        
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        
        stride = 5 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        
        # Shader betöltése fájlból
        try:
            self.program = create_shader_program("shaders/sunglow.vert", "shaders/sunglow.frag")
        except Exception as e:
            print(f"Hiba a SunGlow shader betöltésekor: {e}")
            sys.exit()

    def draw(self, view, projection):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE) 
        gl.glDepthMask(gl.GL_FALSE) 
        
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))
        
        model = glm.mat4(1.0) 
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "model"), 1, gl.GL_FALSE, glm.value_ptr(model))
        
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        
        gl.glDepthMask(gl.GL_TRUE)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

class AsteroidBelt:
    def __init__(self, count, min_radius, max_radius, texture_id, sphere_mesh):
        self.count = count
        self.texture_id = texture_id
        self.sphere = sphere_mesh
        
        # Mátrixok generálása
        self.model_matrices = np.zeros(count * 16, dtype=np.float32)
        for i in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(min_radius, max_radius)
            height = random.uniform(-0.5, 0.5)
            scale = random.uniform(0.05, 0.15)
            
            x = math.cos(angle) * dist
            z = math.sin(angle) * dist
            
            model = glm.mat4(1.0)
            model = glm.translate(model, glm.vec3(x, height, z))
            rot_angle = random.uniform(0, 360)
            model = glm.rotate(model, glm.radians(rot_angle), glm.vec3(random.random(), random.random(), random.random()))
            model = glm.scale(model, glm.vec3(scale))
            
            flat_model = np.array(model.to_list(), dtype=np.float32).flatten()
            self.model_matrices[i*16 : (i+1)*16] = flat_model

        # Buffer beállítása
        gl.glBindVertexArray(self.sphere.vao)
        self.instance_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.model_matrices.nbytes, self.model_matrices, gl.GL_STATIC_DRAW)
        
        # Mat4 beállítása 
        start_loc = 3 
        stride = 16 * 4
        for i in range(4):
            gl.glEnableVertexAttribArray(start_loc + i)
            gl.glVertexAttribPointer(start_loc + i, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(i * 16))
            gl.glVertexAttribDivisor(start_loc + i, 1) # Fontos: példányonkénti léptetés
            
        gl.glBindVertexArray(0)

        # Shader betöltése fájlból
        try:
            self.program = create_shader_program("shaders/asteroid.vert", "shaders/asteroid.frag")
        except Exception as e:
            print(f"Hiba az Asteroid shader betöltésekor: {e}")
            sys.exit()

    def draw(self, view, projection):
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))
        gl.glUniform3fv(gl.glGetUniformLocation(self.program, "lightPos"), 1, glm.value_ptr(glm.vec3(0,0,0)))
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        
        gl.glBindVertexArray(self.sphere.vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.sphere.indices), gl.GL_UNSIGNED_INT, None, self.count)
        gl.glBindVertexArray(0)

class HUD:
    def __init__(self, display_width, display_height):
        pygame.font.init(); self.font = pygame.font.SysFont(None, 22)
        try: self.shader = create_shader_program("shaders/hud.vert", "shaders/hud.frag")
        except Exception as e: print(f"Hiba a HUD shader betöltésekor: {e}"); pygame.quit(); sys.exit()
        self.vao = gl.glGenVertexArrays(1); self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 6 * 4 * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0); gl.glBindVertexArray(0)
        self.update_projection(display_width, display_height)
    def update_projection(self, width, height):
        self.projection = glm.ortho(0.0, float(width), float(height), 0.0)
    def draw(self, text, x, y, color=(255, 255, 255, 255)):
        # 3D effektek kikapcsolása
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        gl.glUseProgram(self.shader)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(self.projection))
        
        surf = self.font.render(text, True, color)
        tex_data = pygame.image.tostring(surf, "RGBA", False) 
        w, h = surf.get_size()
        
        tex = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1) 

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, tex_data)
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        vertices = np.array([
            x, y + h, 0, 1,
            x, y, 0, 0,
            x + w, y, 1, 0,
            x, y + h, 0, 1,
            x + w, y, 1, 0,
            x + w, y + h, 1, 1
        ], dtype=np.float32)
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        
        gl.glDeleteTextures([tex])
        gl.glDisable(gl.GL_BLEND)
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

# --- Főprogram ---
def main():
    pygame.init()
    
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4) 

    # Képernyő méret lekérdezése
    display_info = pygame.display.Info()
    display = (display_info.current_w, display_info.current_h)
    
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL | FULLSCREEN, vsync=1)
    except TypeError:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL | FULLSCREEN)

    pygame.display.set_caption("Modern OpenGL Naprendszer Csillagokkal")
    
    gl.glEnable(gl.GL_MULTISAMPLE)
    
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glCullFace(gl.GL_BACK)
    gl.glViewport(0, 0, display[0], display[1])
    try:
        planet_shader = create_shader_program("shaders/planet.vert", "shaders/planet.frag")
        sun_shader = create_shader_program("shaders/planet.vert", "shaders/sun.frag")
        star_shader = create_shader_program("shaders/star.vert", "shaders/star.frag") 
        orbit_shader = create_shader_program("shaders/orbit.vert", "shaders/orbit.frag") 
        sun_tex = load_texture("textures/sun.jpg"); mercury_tex = load_texture("textures/mercury.jpg")
        venus_tex = load_texture("textures/venus.jpg"); earth_tex = load_texture("textures/earth.jpg")
        mars_tex = load_texture("textures/mars.jpg"); jupiter_tex = load_texture("textures/jupiter.jpg")
        saturn_tex = load_texture("textures/saturn.jpg"); uranus_tex = load_texture("textures/uranus.jpg")
        neptune_tex = load_texture("textures/neptune.jpg"); saturn_ring_tex = load_texture("textures/saturn_ring.png")
        moon_tex = load_texture("textures/moon.jpg") 
    except Exception as e:
        print(f"Hiba a betöltés közben: {e}"); print("Ellenőrizd, hogy a 'shaders' és 'textures' mappák léteznek-e."); return

    hud = HUD(display[0], display[1])
    sphere_mesh = Sphere()
    ring_mesh = Ring(1.2, 2.0)
    star_field = StarField(num_stars=20000, spread=2000.0)

    sun = Planet("Nap", 3.0, 0, 0, 0.5, sun_tex)
    planets = [
        Planet("Merkúr", 0.4, 6, 4.7, 2.5, mercury_tex), 
        Planet("Vénusz", 0.8, 9, 3.5, 1.8, venus_tex),
        Planet("Föld", 1.0, 14, 3.0, 2.0, earth_tex, has_moon=True), 
        Planet("Mars", 0.7, 19, 2.4, 1.6, mars_tex),
        Planet("Jupiter", 2.2, 26, 1.3, 3.0, jupiter_tex),
        Planet("Szaturnusz", 1.9, 34, 0.96, 2.7, saturn_tex, has_rings=True, ring_texture_id=saturn_ring_tex),
        Planet("Uránusz", 1.4, 41, 0.68, 2.5, uranus_tex), 
        Planet("Neptunusz", 1.3, 47, 0.54, 2.3, neptune_tex)
    ]
    moon = Planet("Hold", 0.3, 0, 0.0, 1.5, moon_tex) 
    orbits = [Orbit(planet.distance) for planet in planets if planet.distance > 0]

    sun_glow = SunGlow(size=9.0)
    asteroid_belt = AsteroidBelt(2000, 20.5, 25.0, moon_tex, sphere_mesh)

    camera_pos = glm.vec3(0, 20, 70); camera_front = glm.vec3(0, 0, -1); camera_up = glm.vec3(0, 1, 0)
    yaw, pitch = -90.0, -20.0; speed_factor = 1.0; paused = False
    last_time = pygame.time.get_ticks(); fps = 0
    pygame.mouse.set_visible(False); pygame.event.set_grab(True)

    running = True
    while running:
        current_time = pygame.time.get_ticks(); delta_time = (current_time - last_time) / 1000.0; last_time = current_time
        if delta_time > 0: fps = 1.0 / delta_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE): running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE: paused = not paused
                elif event.key in (K_EQUALS, K_PLUS, K_KP_PLUS): speed_factor *= 1.2
                elif event.key in (K_MINUS, K_UNDERSCORE, K_KP_MINUS): speed_factor /= 1.2
            elif event.type == VIDEORESIZE: 
                display = (event.w, event.h)
                gl.glViewport(0, 0, display[0], display[1])
                hud.update_projection(display[0], display[1])

        cam_speed = 30 * delta_time
        keys = pygame.key.get_pressed()
        if keys[K_w]: camera_pos += cam_speed * camera_front
        if keys[K_s]: camera_pos -= cam_speed * camera_front
        if keys[K_a]: camera_pos -= glm.normalize(glm.cross(camera_front, camera_up)) * cam_speed
        if keys[K_d]: camera_pos += glm.normalize(glm.cross(camera_front, camera_up)) * cam_speed
        if keys[K_e]: camera_pos += cam_speed * camera_up
        if keys[K_q]: camera_pos -= cam_speed * camera_up

        mx, my = pygame.mouse.get_rel(); sensitivity = 0.1
        yaw += mx * sensitivity; pitch -= my * sensitivity; pitch = max(-89.0, min(89.0, pitch))
        front = glm.vec3()
        front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
        front.y = math.sin(glm.radians(pitch)); front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
        camera_front = glm.normalize(front)

        if not paused:
            sun.update(delta_time * 50 * speed_factor)
            for planet in planets: planet.update(delta_time * 50 * speed_factor)

        gl.glClearColor(0.0, 0.0, 0.05, 1.0); gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        projection = glm.perspective(glm.radians(45), display[0] / display[1], 0.1, 5000.0)
        view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)

        # Csillagmező 
        gl.glUseProgram(star_shader)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(star_shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(star_shader, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        star_field.draw()
        
        # Nap (sun)
        gl.glUseProgram(sun_shader)
        loc = gl.glGetUniformLocation(sun_shader, "projection"); gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(projection))
        loc = gl.glGetUniformLocation(sun_shader, "view"); gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(view))
        model = sun.get_model_matrix()
        loc = gl.glGetUniformLocation(sun_shader, "model"); gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(model))
        gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, sun.texture_id)
        sphere_mesh.draw()

        sun_glow.draw(view, projection)

        # Bolygók, Hold és Gyűrűk
        gl.glUseProgram(planet_shader)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(planet_shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(planet_shader, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniform3fv(gl.glGetUniformLocation(planet_shader, "lightPos"), 1, glm.value_ptr(glm.vec3(0,0,0)))
        gl.glUniform3fv(gl.glGetUniformLocation(planet_shader, "viewPos"), 1, glm.value_ptr(camera_pos))

        for planet in planets:
            model = planet.get_model_matrix()
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(planet_shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(model))
            gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, planet.texture_id)
            sphere_mesh.draw()

            if planet.has_moon:
                base_model = planet.get_model_matrix(scale_override=1.0)
                moon_model_matrix = glm.rotate(base_model, glm.radians(planet.moon_angle), glm.vec3(0.2, 1, 0))
                moon_model_matrix = glm.translate(moon_model_matrix, glm.vec3(planet.radius * 2.5, 0, 0))
                moon_model_matrix = glm.scale(moon_model_matrix, glm.vec3(moon.radius)) 
                gl.glUniformMatrix4fv(gl.glGetUniformLocation(planet_shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(moon_model_matrix))
                gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, moon.texture_id) 
                sphere_mesh.draw()

            if planet.has_rings:
                gl.glEnable(gl.GL_BLEND); gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                base_model = planet.get_model_matrix(scale_override=1.0)
                ring_model_matrix = glm.rotate(base_model, glm.radians(75), glm.vec3(1, 0, 0))
                ring_model_matrix = glm.scale(ring_model_matrix, glm.vec3(planet.radius))
                gl.glUniformMatrix4fv(gl.glGetUniformLocation(planet_shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(ring_model_matrix))
                gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, planet.ring_texture_id)
                ring_mesh.draw()
                gl.glDisable(gl.GL_BLEND)

        asteroid_belt.draw(view, projection)

        gl.glUseProgram(orbit_shader)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(orbit_shader, "projection"), 1, gl.GL_FALSE, glm.value_ptr(projection))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(orbit_shader, "view"), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniform3fv(gl.glGetUniformLocation(orbit_shader, "orbitColor"), 1, glm.value_ptr(glm.vec3(1.0, 1.0, 1.0))) # Tiszta fehér szín
        model = glm.mat4(1.0)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(orbit_shader, "model"), 1, gl.GL_FALSE, glm.value_ptr(model))
        for orbit in orbits:
            orbit.draw()

        # HUD kirajzolása
        hud_text = f"FPS: {int(fps)}   Sebesség: {speed_factor:.2f}x   {'[SZÜNET]' if paused else ''}   Kezelés: WASDQE + Egér"
        hud.draw(hud_text, 10, 10)

        pygame.display.flip()
    pygame.quit()

if __name__ == '__main__':
    main()