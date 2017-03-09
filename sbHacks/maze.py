# -*- coding: utf-8 -*-
"""
Purpose: Reduce a picture of a maze to a graph, and navigate 
         the graph with a reinforcement learning algorithm.
         Done as a last minute idea for SB Hacks 2016.
         
Author:  David Rower
Date:    January 2017
"""

print(__doc__)

from scipy import misc
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import string

def getDims(img):
    """  Finds number of unique columns, rows (N,M) in image (n,m). """
    n,m,k = np.shape(img) 
    N,M   = 0,0
    for i in range(1,n):
        if np.array_equal(img[i],img[i-1]):
            N += 1
    for j in range(1,m):
        if np.array_equal(img[:,j],img[:,j-1]):
            M += 1
    return N,M,n,m

def reduceImage(img,N,M,n,m):
    """ Turns original (n,m) image into reduced (N,M) image. """
    scaleN = int(n/(2*N))
    scaleM = int(m/(2*M))
    imgR = np.zeros((2*N+1,2*M+1))
    for i in range(2*N+1):
        for j in range(2*M+1):
            if img[i*scaleN+2,j*scaleM+2,3] != 255:
                imgR[i,j] = 0.
            else: 
                imgR[i,j] = 1.
    return imgR

def adjPaths(imgR,location):
    """ Finds possible path in maze from a certain location. """
    directions = [(1,0),(-1,0),(0,1),(0,-1)] # up, down, left, right 
    possiblePaths = []                               
    for direction in directions:
        iPlus,jPlus = direction
        if imgR[location[0]+iPlus,location[1]+jPlus] == 0: 
            possiblePaths.append(direction)
    return possiblePaths
    
def placeNodes(imgR):
    """ Finds locations in maze where path branches. """
    nodes = []
    N,M = np.shape(imgR)
    for i in range(N):
        for j in range(M):
            loc = (i,j)
            if imgR[loc] == 0.:
                if len(adjPaths(imgR,loc)) > 2:
                    nodes.append(loc)
    return nodes 

def step(node,path):
    """ Steps in certain direction from a node. """
    return (node[0]+path[0],node[1]+path[1])
    
def checkPath(imgR,nodes,node,path):
    """ Determines if nodes connect to other nodes via a certain path. """
    newLoc = step(node,path)
    adjPathList = adjPaths(imgR,newLoc)
    l = len(adjPathList)
    if l == 1:
        return 0
    if l == 2: 
        openPath = True
        while openPath:
            adjPathList.remove((-path[0],-path[1]))
            path   = adjPathList[0]
            newLoc = step(newLoc,path)
            newL   = len(adjPaths(imgR,newLoc))
            if newL == 1: # we have reached a dead end
                return 0
            elif newL == 2: # step again
                adjPathList = adjPaths(imgR,newLoc)
            elif newL > 2: # we have found another node
                return newLoc
    
def connectNodes(imgR,nodes,start,goal):
    """ Generates graph representing the maze. """
    alphabet = string.ascii_lowercase
    nodeConnections = [[] for i in range(len(nodes)+2)]
    for index, node in enumerate(nodes):
        paths = adjPaths(imgR,node)
        for path in paths:
            result = checkPath(imgR,nodes,node,path)
            if not result == 0: 
                nIndex = nodes.index(result)
                nodeConnections[index+1].append(alphabet[nIndex+1])
    paths = adjPaths(imgR,start)  # add start to nodes
    for path in paths:
            result = checkPath(imgR,nodes,start,path)
            if not result == 0: 
                nIndex = nodes.index(result)
                nodeConnections[0].append(alphabet[nIndex+1])
    for node in nodeConnections[0]:
        nodeConnections[alphabet.index(node)].append(alphabet[0])
    paths = adjPaths(imgR,goal)  # add goal to nodes
    for path in paths:
            result = checkPath(imgR,nodes,goal,path)
            if not result == 0: 
                nIndex = nodes.index(result)
                nodeConnections[len(nodeConnections)-1].append(alphabet[nIndex+1])
    for node in nodeConnections[len(nodeConnections)-1]:
        nodeConnections[alphabet.index(node)].append(alphabet[len(nodeConnections)-1])
    return [alphabet[i] for i in range(len(nodes)+2)], nodeConnections

def generateR(nodes,nodeConnections):
    R = -1*np.ones((len(nodeConnections),len(nodeConnections)))
    for stateIndex in range(len(nodes)): 
        for actionIndex in range(len(nodes)):
            if nodes[actionIndex] in nodeConnections[stateIndex]:
                R[stateIndex,actionIndex] = 0.
                if nodes[actionIndex] == nodes[len(nodes)-1]:
                    R[stateIndex,actionIndex] = 100.
    return R

def updateQ(Q,R,gamma,state,action):
    Q[state,action] = R[state,action] + gamma * max(Q[action])
    Q[state][Q[state] > 0] /= np.sum(Q[state][Q[state] > 0])
    return Q
    
def initializeParams(N,nodes,nodeConnections):
    trials = 1000
    gamma = 0.01
    Q = np.zeros((N,N))
    R = generateR(nodes,nodeConnections)
    return trials, gamma, Q, R
    
def runEpisode(gamma,Q,R,nodes,nodeConnections,start,goal): 
    currentState = nodes.index(start)
    finalState   = nodes.index(goal)
    counter = 0
    while currentState != finalState: 
        counter += 1
        try:
            if np.sum(q[current_state]) > 0:
                action = np.argmax(Q[currentState])
        except:
            action = np.random.randint(0,len(nodes))
        Q  = updateQ(Q,R,gamma,currentState,action)
        currentState = action
        if currentState == finalState:
            break
    return Q, counter

def main():
    imgPath = 'maze.png'
    print('Reading %s'%imgPath)
    img   = misc.imread(imgPath)  
    N,M,n,m = getDims(img)
    N,M   = 13,13
    print('Reducing %s'%imgPath)
    imgR = reduceImage(img,N,M,n,m)
    imgR[0,1]  = 1.
    imgR[26,3] = 1.
    print('Plotting original, reduced image.')
    plt.subplot(121),plt.imshow(img,  cmap = "Greys", interpolation = "nearest")
    plt.title('Original Image')
    plt.subplot(122),plt.imshow(imgR, cmap = "Greys", interpolation = "nearest")
    plt.title('Reduced image')
    plt.show()
    print('')

    start = (1,1)
    goal  = (25,3)
    print('Finding nodes in maze.')
    nodes = placeNodes(imgR)
    nodeLocations = nodes.copy()
    print('Nodes: ', nodes)
    print('')

    print('Finding node connections in maze.')
    nodes, nodeConnections = connectNodes(imgR,nodes,start,goal)
    print('Node Connections: ',nodeConnections)
    print('')

    print('Plotting reduced image with node locations.')
    plt.imshow(imgR, cmap = "Greys", interpolation = "nearest")
    alphabet = string.ascii_lowercase
    for index, node in enumerate(nodeLocations):
        plt.text(node[1],node[0],alphabet[index])
    plt.title('Reduced image with nodes')
    plt.show()

    print('Starting Q-learning algorithm.')
    nodeN = len(nodes)
    start,goal = nodes[0],nodes[nodeN-1]
    trials, gamma, Q, R = initializeParams(nodeN,nodes,nodeConnections)
    counters = []
    for trial in range(trials): 
        Q, counter = runEpisode(gamma,Q,R,nodes,nodeConnections,start,goal)
#        print("Trial %s: %s steps"%(trial,counter))
        counters.append(counter)
    print('Plotting results.')
    plt.figure()
    plt.plot(range(trials),counters)
    plt.xlabel('Trial number')
    plt.ylabel('Number of steps to reach end')
    plt.show()
    


if __name__ == "__main__": 
    main()
    
    
