import math
import time
from dataclasses import dataclass
import random

import pygame

sw, sh = 200, 200
tw, th = 50, 50
pw, ph = sw/tw, sh/th
pygame.init()
screen = pygame.display.set_mode((sw, sh))
color = [0, 0, 0]

def drawPixel(x, y):
    pygame.draw.rect(screen, color, [x*pw, y*ph, pw, ph])

def setColor(r, g, b):
    global color
    color = [r, g, b]


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __mul__(self, scalar):
        if type(scalar) in [int, float]:
            return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
        elif type(scalar) == Vec3:
            return Vec3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
        elif type(scalar) == Color:
            return Color(self.x * scalar.r, self.y * scalar.g, self.z * scalar.b)

    def __sub__(self, scalar):
        if type(scalar) in [int, float]:
            return Vec3(self.x - scalar, self.y - scalar, self.z - scalar)
        elif type(scalar) == Vec3:
            return Vec3(self.x - scalar.x, self.y - scalar.y, self.z - scalar.z)
        elif type(scalar) == Color:
            return Color(self.x - scalar.r, self.y - scalar.g, self.z - scalar.b)

    def __add__(self, scalar):
        if type(scalar) in [int, float]:
            return Vec3(self.x + scalar, self.y + scalar, self.z + scalar)
        elif type(scalar) == Vec3:
            return Vec3(self.x + scalar.x, self.y + scalar.y, self.z + scalar.z)
        elif type(scalar) == Color:
            return Color(self.x + scalar.r, self.y + scalar.g, self.z + scalar.b)

    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def set_magnitude(self, magnitude):
        multiplier = magnitude / self.magnitude()
        x = multiplier * self.x
        y = multiplier * self.y
        z = multiplier * self.z
        return Vec3(x, y, z)

@dataclass
class Color:
    r: float
    g: float
    b: float
    
    def __mul__(self, scalar):
        if type(scalar) in [int, float]:
            return Color(self.r * scalar, self.g * scalar, self.b * scalar)
        elif type(scalar) == Vec3:
            return Color(self.r * scalar.x, self.g * scalar.y, self.b * scalar.z)
        elif type(scalar) == Color:
            return Color(self.r * scalar.r, self.g * scalar.g, self.b * scalar.b)

    def __sub__(self, scalar):
        if type(scalar) in [int, float]:
            return Color(self.r - scalar, self.g - scalar, self.b - scalar)
        elif type(scalar) == Vec3:
            return Color(self.r - scalar.x, self.g - scalar.y, self.b - scalar.z)
        elif type(scalar) == Color:
            return Color(self.r - scalar.r, self.g - scalar.g, self.b - scalar.b)

    def __add__(self, scalar):
        if type(scalar) in [int, float]:
            return Color(self.r + scalar, self.g + scalar, self.b + scalar)
        elif type(scalar) == Vec3:
            return Color(self.r + scalar.x, self.g + scalar.y, self.b + scalar.z)
        elif type(scalar) == Color:
            return Color(self.r + scalar.r, self.g + scalar.g, self.b + scalar.b)
grid = [[Color(0, 0, 0) for i in range(sw)] for j in range(sh)]
@dataclass
class Material:
    smoothness: float
    color: Color
    lightColor: Color
    lightStrength: float

@dataclass
class Sphere:
    position: Vec3
    radius: float
    material: Material

@dataclass
class Ray:
    origin: Vec3
    direction: Vec3

@dataclass
class HitInfo:
    didHit: bool
    distance: float
    normal: Vec3
    hitPoint: Vec3
    material: Material
defaultMaterial = Material(0, Color(1.0, 0, 0), Color(0, 0, 0), 0)
defaultLight = Material(0, Color(0, 1.0, 0), Color(1.0, 1.0, 1.0), 1)
emptyMaterial = Material(0, Color(0, 0, 0), Color(0, 0, 0), 0)
spheres = [
    Sphere(Vec3(0, 0, 30), 5, defaultMaterial),
    Sphere(Vec3(-5, 10, 20), 5, defaultLight),
]

def sign(value):
    if value < 0:
        return -1
    return 1

def dot(vec1: Vec3, vec2: Vec3):
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z

def normalize(vec: Vec3):
    return vec.set_magnitude(1)

def randomSphere():
    rotation = random.random() * math.pi * 2
    zRotation = random.random() * math.pi * 2
    circleSize = math.cos(zRotation)
    xCord = circleSize*math.cos(rotation)
    yCord = circleSize*math.sin(rotation)
    zCord = math.sin(zRotation)
    return Vec3(xCord, yCord, zCord)

def randomHemisphere(normal: Vec3):
    direction = randomSphere()
    return direction * sign(dot(normal, direction))

def raySphereCollision(ray: Ray, sphere: Sphere) -> HitInfo:
    info: HitInfo = HitInfo(False, -1.0, Vec3(0, 0, 0), Vec3(0, 0, 0), emptyMaterial);
    oc = ray.origin - sphere.position
    a = dot(ray.direction, ray.direction)
    b = 2.0 * dot(oc, ray.direction)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4.0 * a * c
    #print(ray.direction)
    if discriminant >= 0.0:
        distance = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if distance >= 0.0:
            info.didHit = True
            info.distance = distance
            info.hitPoint = ray.origin + ray.direction * distance
            info.normal = normalize(info.hitPoint - sphere.position)
    return info


def rayIntersection(ray: Ray):
    closest: HitInfo = HitInfo(False, -1.0, Vec3(0, 0, 0), Vec3(0, 0, 0), emptyMaterial);
    for i in range(len(spheres)):
        result = raySphereCollision(ray, spheres[i])
        if result.didHit and (not closest.didHit or result.distance < closest.distance):
            closest = result
            closest.material = spheres[i].material
    return closest


def mix(vec1, vec2, inbetween):
    colorDiffrence = vec2-vec1
    return vec1+colorDiffrence*inbetween


def reflect(direction, normal):
    return direction - normal * dot(normal, direction)


def copyVec3(vec):
    return Vec3(vec.x, vec.y, vec.z)


def copyRay(ray):
    return Ray(copyVec3(ray.orgin), copyVec3(ray.direction))


def traceRay(ray: Ray, depth = 0, rayColor: Color = Color(1, 1, 1)) -> Color:
    incomingLight: Color = Color(0, 0, 0)
    original = copyRay(ray)
    if depth<2:
        hit: HitInfo = rayIntersection(ray)
        
        if hit.didHit:
            #print(hit.distance)
            material = hit.material
            emittedLight = material.lightColor * material.lightStrength
            incomingLight += emittedLight * rayColor
            rayColor *= material.color
            combindedColor = Color(0, 0, 0)
            for i in range(10):
                ray = copyRay(original)
                ray.origin = hit.hitPoint
                dif = randomHemisphere(hit.normal)
                ref = ray.direction = reflect(ray.direction, hit.normal)
                ray.direction = mix(dif, ref, material.smoothness)
                combindedColor = combindedColor+
            combindedColor = combindedColor * 0.1
            incomingLight += combindedColor
            #print(incomingLight)
        else:
            #rayColor *= ambientColor
            pass
    #if incomingLight.magnitude <= 0.01:
    #    incomingLight += ambientLight * rayColor * ambientStrength;
    return incomingLight


def colorCalculation(x, y):
    result = Color(0, 0, 0)
    ray = Ray(Vec3(0, 0, 0), normalize(Vec3(x-0.5, y-.5, 1)))
    result = result + traceRay(ray)
    #result = result*0.03
    result.r = min(result.r, 1)
    result.g = min(result.g, 1)
    result.b = min(result.b, 1)

    #print(result)
    return [result.r, result.g, result.b]
def setPixelColors(func):
    global grid
    for i in range(th):
        for j in range(tw):
            result = func(j/tw, i/th)
            grid[i][j] = (grid[i][j]+Color(result[0], result[1], result[2]))*0.5
            setColor(grid[i][j].r*255, grid[i][j].g*255, grid[i][j].b*255)
            drawPixel(j, i)
            #pygame.display.update()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #pygame.draw.rect(screen, (255,255,255), (0, 0, sw, sh))
    #pygame.display.update()
    setPixelColors(colorCalculation)

    pygame.display.update()
    #running = False
    #time.sleep(1)

pygame.quit()