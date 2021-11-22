import numpy as np
import cv2


def order_points(pts) -> np.ndarray:
    # Ordenamos los puntos de la imagen en una matriz de 4x2
    # 1ª top-left, 2ª top-right, 3ª bottom-right, 4ª bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # De los puntos que nos dan, el que tenga la menor suma de sus coordenadas
    # es el top-left, el que tenga la mayor suma de sus coordenadas es el bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Ahora, el top-right es el que tenga la resta de las coordenadas más grande
    # y el bottom-left es el que tenga la resta de las coordenadas más pequeña
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


# ----------------------------------------------------------------------------- #
def four_point_transform(image, pts) -> np.ndarray:
    # Obtenemos los punts ordenados
    rect = order_points(pts)

    # Hacemos un unpack de los puntos para tratarlos individualmente
    (tl, tr, br, bl) = rect

    # Calculamos la altura y anchura de la nueva imagen

    # El ancho será el máximo entre br-bl y tr-tl
    widthA = two_points_distance(br, bl)
    widthB = two_points_distance(tr, tl)
    maxWidth = max(int(widthA), int(widthB))

    # La altura será el máximo entre tr-br y tl-bl
    heightA = two_points_distance(tr, br)
    heightB = two_points_distance(tl, bl)
    maxHeight = max(int(heightA), int(heightB))

    # Construyo la matriz final, con los cuatro puntos finales,
    # una vez que tengo el alto y el ancho
    dst = np.array([
        [0, 0]
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Calculamos la matriz de transformación
    M = cv2.getPerspectiveTransform(rect, dst)

    image_mod = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return image_mod


# ----------------------------------------------------------------------------- #
def two_points_distance(point1: tuple, point2: tuple) -> float:
    # Calculamos la distancia entre dos puntos
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
