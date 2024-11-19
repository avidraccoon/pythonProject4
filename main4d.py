import math
import time
from dataclasses import dataclass
import random
import numpy as np

import pygame
from numpy import ndarray

sw, sh = 800, 800
tw, th = 200, 200
pw, ph = sw/tw, sh/th
pygame.init()
screen = pygame.display.set_mode((sw, sh))
color = [0, 0, 0]
frames = 1
def drawPixel(x, y):
    #print(color)
    pygame.draw.rect(screen, color, [x*pw, y*ph, pw, ph])

def setColor(r, g, b):
    global color
    color = [r, g, b]


grid = [[np.array([-1.0, 0, 0]) for i in range(sw)] for j in range(sh)]
@dataclass
class Material:
    smoothness: float
    color: ndarray
    lightColor: ndarray
    lightStrength: float

@dataclass
class Sphere:
    position: ndarray
    radius: float
    material: Material

@dataclass
class Ray:
    origin: ndarray
    direction: ndarray

@dataclass
class HitInfo:
    didHit: bool
    distance: float
    normal: ndarray
    hitPoint: ndarray
    material: Material
defaultMaterial = Material(0, np.array([1.0, 0, 0]), np.array([0.0, 0, 0]), 0)
defaultMaterial2 = Material(0, np.array([0.7, 0.7, 0.2]), np.array([0.0, 0, 0]), 0)
defaultMaterial3 = Material(0, np.array([0, 0, 1]), np.array([0.0, 0, 0]), 0)
defaultLight = Material(0, np.array([1, 1, 1]), np.array([1, 1, 1]), 60)
defaultLight2 = Material(0, np.array([1, 1, 1]), np.array([1, 1, 1]), 2)
emptyMaterial = Material(0, np.array([1, 1, 1]), np.array([0.0, 0, 0]), 0)
"""spheres = [
    Sphere(np.array([-2, 1, 30]), 5, defaultMaterial),
    #Sphere(np.array([-6, 1, 30]), 5, defaultMaterial3),
    Sphere(np.array([0, 30, 30]), 25, defaultMaterial2),
    Sphere(np.array([15, -50, 80]), 15, defaultLight),
    Sphere(np.array([-15, 30, -50]), 15, defaultLight2),
]"""

spheres = [
    Sphere(np.array([0, 8, 50, 8]), 10, defaultMaterial),
    Sphere(np.array([0, -8, 50, 8]), 10, defaultLight),
]

def sign(value):
    if value < 0:
        return -1
    return 1

def dot(vec1: ndarray, vec2: ndarray):
    return vec1.dot(vec2)

def normalize(vec: ndarray):
    return vec / np.linalg.norm(vec)

def randomSphere():
    rotation = random.random() * math.pi * 2
    zRotation = random.random() * math.pi * 2
    circleSize = math.cos(zRotation)
    xPos = circleSize*math.cos(rotation)
    yPos = circleSize*math.sin(rotation)
    zPos = math.sin(zRotation)
    return np.array([xPos, yPos, zPos, circleSize])

def randomHemisphere(normal: ndarray):
    direction = randomSphere()
    return direction * sign(dot(normal, direction))

def raySphereCollision(ray: Ray, sphere: Sphere) -> HitInfo:
    info: HitInfo = HitInfo(False, -1.0, np.array([0.0, 0, 0]), np.array([0.0, 0, 0]), emptyMaterial);
    """oc = ray.origin - sphere.position
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
            """
    V = sphere.position - ray.origin
    bb = dot(V, ray.direction)
    rad = (bb*bb) - dot(V, V) + sphere.radius*sphere.radius
    #print(rad)
    if rad < 0:
        return info

    rad = math.sqrt(rad)
    t2 = bb - rad
    t1 = bb + rad

    #print(t1, t2)

    if t1 < 0 or (t2 > 0 and t2 < t1):
        t1 = t2
    if t1 < 0:
        return info

    intersect = ray.origin + ray.direction * t1
    normal = (intersect - sphere.position) / sphere.radius
    info.normal = normal
    info.hitPoint = ray.origin + ray.direction * t1
    info.didHit = True
    info.distance = np.linalg.norm(ray.direction * t1)

    #print(info.distance)
    return info


def rayIntersection(ray: Ray):
    closest: HitInfo = HitInfo(False, -1.0, np.array([0.0, 0, 0]), np.array([0.0, 0, 0]), emptyMaterial)
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
    return direction - normal * dot(normal, direction) * 2


def copyVec3(vec:ndarray):
    return np.array([vec[0], vec[1], vec[2], vec[3]])


def copyRay(ray):
    return Ray(copyVec3(ray.origin), copyVec3(ray.direction))

target_dep = 2
def traceRay(ray: Ray, depth = 0, rayColor = None) -> ndarray:
    if type(rayColor) == type(None): rayColor = np.array([1.0, 1, 1])
    incomingLight: ndarray = np.array([0.0, 0, 0])
    original = copyRay(ray)
    if depth<target_dep:
        hit: HitInfo = rayIntersection(ray)
        
        if hit.didHit:
            #print(hit.distance)
            #hit.normal += 1.0
            #hit.normal /= 2.0
            #return np.array([hit.normal[0], hit.normal[1], hit.normal[2]])
            material = hit.material
            emittedLight = material.lightColor * material.lightStrength
            incomingLight += emittedLight * rayColor
            rayColor *= material.color
                #pass
            #return rayColor
            #if depth < 2:
            combindedColor = np.array([0.0, 0.0, 0.0])
            for i in range(5):
                ray = copyRay(original)
                ray.origin = hit.hitPoint
                dif = randomHemisphere(hit.normal)
                ref = reflect(ray.direction, hit.normal)
                ray.direction = mix(dif, ref, material.smoothness)
                combindedColor = combindedColor+traceRay(ray, depth+1, rayColor)
            #return combindedColor
            combindedColor = combindedColor * 0.2
            incomingLight += combindedColor
            #print(incomingLight)


        else:
            #rayColor *= ambientColor
            pass
    #return rayColor
    #
    #if incomingLight.magnitude <= 0.01:
    #    incomingLight += ambientLight * rayColor * ambientStrength;
    #print(depth)
    return incomingLight


def colorCalculation(x, y):
    result = np.array([0.0, 0, 0])
    for i in range(5):
        ray = Ray(np.array([0.0, 0, 0, 0]), normalize(np.array([x-0.5, y-0.5, 1, frames/100])))
        result = result + traceRay(ray)
    result = result * 0.2
    result[0] = min(result[0], 1)
    result[1] = min(result[1], 1)
    result[2] = min(result[2], 1)

    #print(result)
    return [result[0], result[1], result[2]]
def setPixelColors(func):
    global grid, frames
    prev_multi = 1-(1/math.sqrt(frames))
    cur_multi = (1/math.sqrt(frames))
    for i in range(th):
        for j in range(tw):
            result = func(j/tw, i/th)
            #if frames == 1:
            grid[i][j] = np.array([result[0], result[1], result[2]])
            #else:
            #    grid[i][j] = (grid[i][j]*prev_multi+np.array([result[0], result[1], result[2]])*cur_multi)
            #if grid[i][j][0] > 0.1: print(grid[i][j], result)
            setColor(grid[i][j][0]*255, grid[i][j][1]*255, grid[i][j][2]*255)
            drawPixel(j, i)
        pygame.display.update()
        #print(i)
    print("Frame: "+str(frames)+" Multi: "+str(cur_multi))
    frames+=1
running = True

while running:
    m=0.1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                target_dep -= 3
                frames = 1
                print(target_dep)
            if event.key == pygame.K_RIGHT:
                target_dep += 3
                frames = 1
                print(target_dep)
            if event.key == pygame.K_1:
                defaultMaterial.smoothness-=m
                frames = 1
                print(defaultMaterial.smoothness)
            if event.key == pygame.K_2:
                defaultMaterial.smoothness +=m
                frames = 1
                print(defaultMaterial.smoothness)
            if event.key == pygame.K_3:
                defaultMaterial2.smoothness-=m
                frames = 1
                print(defaultMaterial2.smoothness)
            if event.key == pygame.K_4:
                defaultMaterial2.smoothness +=m
                frames = 1
                print(defaultMaterial2.smoothness)
            if event.key == pygame.K_5:
                defaultMaterial3.smoothness-=m
                frames = 1
                print(defaultMaterial3.smoothness)
            if event.key == pygame.K_6:
                defaultMaterial3.smoothness +=m
                frames = 1
                print(defaultMaterial3.smoothness)

    #pygame.draw.rect(screen, (255,255,255), (0, 0, sw, sh))
    #pygame.display.update()
    setPixelColors(colorCalculation)

    pygame.display.update()
    #running = False
    #time.sleep(1)


pygame.quit()