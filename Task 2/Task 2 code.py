import cv2
import numpy as np
import math
import heapq
import random

OBSTACLES = []
#Graph object
class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.pos] = node

    def get_node(self, pos):
        return self.nodes.get(pos)

    def get_neighbors(self, node):
        neighbors = node.neighbours
        return neighbors

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.g = math.inf
        self.h = 0
        self.f = math.inf
        self.neighbours=[]
        self.parent = None
    def  add_neighb(self,node):
        self.neighbours.append(node)

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, end):
    return math.sqrt((end[0] - node.pos[0])**2 + (end[1] - node.pos[1])**2)

def A_star(start, end, graph):
    open_set = []
    closed_set = []

    start_node = graph.get_node(start)
    end_node = graph.get_node(end)

    start_node.g = 0
    start_node.h = heuristic(start_node, end_node.pos)
    start_node.f = start_node.g + start_node.h

    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.pos == end_node.pos:
            path = []
            current = current_node
            while current is not None:
                path.append(current.pos)
                current = current.parent
            return path[::-1]

        closed_set.append(current_node)

        for neighbor in current_node.neighbours:
            tent_g = current_node.g + 1

            if tent_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tent_g
                neighbor.h = heuristic(neighbor, end_node.pos)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)




def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return d
# function to find if there is no obstacle between 2 points
def collision_free( from_point, to_point,image):
    line = np.linspace(from_point, to_point, num=50, dtype=int)
    return not np.any([image[y, x] < 128 for x, y in line])
def main():
    maze=cv2.imread('maze.png')


    image=maze.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    rect_coords = [
        ((0, 0), (470, 16)),
        ((0, 0), (8, 355)),
        ((448, 0), (470, 457)),
        ((0, 338), (470, 355))
    ]


    for (x1, y1), (x2, y2) in rect_coords:
        cv2.rectangle(binary_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    img=image.copy()
    h,w,z=image.shape


    start=(162,34)
    end=(412,309)
    start_easy=(44,309)
    end_easy=(106,330)

    cv2.circle(image,(162,34),4,(255,0,0),-1)
    cv2.circle(image,(412,309),4,(0,0,255),-1)
    cv2.circle(image,(44,309),4,(255,0,0),-1)
    cv2.circle(image,(106,330),4,(0,0,255),-1)
        

    # Creating  graph
    graph = Graph()
    start_node=Node(start)
    end_node=Node(end)
    start_easy_node=Node(start_easy)
    end_easy_node=Node(end_easy)
    graph.add_node(start_easy_node)
    graph.add_node(end_easy_node)
    graph.add_node(start_node)
    graph.add_node(end_node)
    for i in range(3000):
        y=random.randint(0, h-1)
        x=random.randint(0, w-1)
        if binary_image[y][x] == 255 :
            node = Node((x, y))
            graph.add_node(node)
            
            cv2.circle(image,(x,y),2,(255,255,0),-1)
    #image showing randomly sampled points
    cv2.imshow('Binary Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    for i in range(w):
        for j in range(h):
            if binary_image[j][i]==0:
                OBSTACLES.append((i,j))

    
    for pos, node in graph.nodes.items():
        nc = [(pos2, node2) for pos2, node2 in graph.nodes.items() if distance(pos, pos2) < 20 and collision_free(pos2, pos, binary_image)]
        nc.sort(key=lambda x: distance(pos, x[0]))  
        node.neighbours = [node2 for _, node2 in nc]  


    path1 = A_star(start, end, graph)#find path between start hard and end hard
    path2=A_star(start_easy,end_easy,graph)#to find path between start easy and end easy
    #visualising the path created
    if path1:
        cv2.line(maze,(163,10),path1[0],(0,0,255),2)
    for i in range(len(path1) - 1):
        cv2.line(maze,path1[i],path1[i+1],(0,0,255),2)

        cv2.imshow('PRM Visualization', maze)
        cv2.waitKey(100)
    if path1:
        cv2.line(maze,(4558,302),path1[len(path1)-1],(0,0,255),2)
    if path2:
        cv2.line(maze,(40,344),path2[0],(0,0,255),2)
    for i in range(len(path2) - 1):
            cv2.line(maze,path2[i],path2[i+1],(0,0,255),2)

            cv2.imshow('PRM Visualization', maze)
            cv2.waitKey(100)
    if path2:
        cv2.line(maze,(97,343),path2[len(path2)-1],(0,0,255),2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()