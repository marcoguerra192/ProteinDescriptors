## Marco Guerra 2025

# Implementation of the dodecahedral symmetry


import numpy as np

# Computations of dodecahedral sectors from a protein point cloud and a centroid

# BUILD THE DODECAHEDRON

# Phi is the golden ratio
phi = (1 + np.sqrt(5)) / 2

a, b = 1, 1 / phi

# 20 vertices of a regular dodecahedron
vertices = np.array([
    # 8 of type (±1, ±1, ±1)
    [+a, +a, +a],
    [+a, +a, -a],
    [+a, -a, +a],
    [+a, -a, -a],
    [-a, +a, +a],
    [-a, +a, -a],
    [-a, -a, +a],
    [-a, -a, -a],
    
    # 4 of type (0, ±1/phi, ±phi)
    [0, +b, +phi],
    [0, +b, -phi],
    [0, -b, +phi],
    [0, -b, -phi],
    
    # 4 of type (±1/phi, ±phi, 0)
    [+b, +phi, 0],
    [+b, -phi, 0],
    [-b, +phi, 0],
    [-b, -phi, 0],
    
    # 4 of type (±phi, 0, ±1/phi)
    [+phi, 0, +b],
    [+phi, 0, -b],
    [-phi, 0, +b],
    [-phi, 0, -b],
])

faces = [
 [0, 8, 10, 2, 16],
 [4, 18, 6, 10, 8],
 [2, 10, 6, 15, 13],
 [0, 12, 14, 4, 8],
 [6, 18, 19, 7, 15],
 [3, 17, 16, 2, 13],
 [0, 16, 17, 1, 12],
 [4, 14, 5, 19, 18],
 [3, 13, 15, 7, 11],
 [1, 9, 5, 14, 12],
 [1, 17, 3, 11, 9],
 [5, 9, 11, 7, 19] ]

# barycenters are the means of the vertices on each face
barycenters = np.vstack( [ np.mean( np.vstack( [ vertices[j,:] for j in fi ] ) , axis = 0) for fi in faces ] )

# Here the vertices of the dodecahedron (and so the circumscribed sphere) have radius sqrt(3)
# and the barycenters (hence the inscribed sphere) have radius 1.376


# Function to order the barycenters
# These ones are already ordered
def order_3D( i ):
    ''' Ordering function: returns the order of the barycenters of the faces:
    first by decreasing z, then counterclockwise.

    '''
    
    v = barycenters[i,:]
    z = -v[2]
    theta = np.arctan2(v[1],v[0])
    
    return ( z , theta )

# Build edges

# edges = set()
# for face in faces:
#     for i in range(5):
#         v1 = face[i]
#         v2 = face[(i + 1) % 5]
#         edges.add(tuple(sorted((v1, v2))))  # this is an undirected edge!

''' Construction of the permutations
'''

# Choose adjacent pairs of vertices to map to

# rotation_pairs = []

# for v1 in range(20): #choose the first vertex

#     # find the adjacent vertices
#     adj_vertices = [ e[1] for e in edges if e[0] == v1 ] + [ e[0] for e in edges if e[1] == v1]

#     for v2 in adj_vertices:

#         rotation_pairs.append( (v1,v2) )

# Function to build the rotation matrix

# def rotation(x, y, p, q):
#     # Normalize to lie on sphere
#     def normalize(v): return v / np.linalg.norm(v)
#     x, y, p, q = map(normalize, (x, y, p, q))

#     # Build source orthonormal basis via Grahm-Schmidt
#     u1 = x
#     u2 = y - np.dot(y, u1) * u1 
#     u2 = normalize(u2)
#     u3 = np.cross(u1, u2)

#     # Build target orthonormal basis
#     v1 = p
#     v2 = q - np.dot(q, v1) * v1
#     v2 = normalize(v2)
#     v3 = np.cross(v1, v2)

#     U = np.stack([u1, u2, u3], axis=1)
#     V = np.stack([v1, v2, v3], axis=1)

#     R = V @ U.T   # rotation matrix

#     return R

## BUILD THE PERMUTATIONS

# permutations = np.zeros((60,12), dtype=int)

# x = vertices[0,:]
# y = vertices[8,:]

# for i, (v1, v2) in enumerate(rotation_pairs):

#     p = vertices[v1,:]
#     q = vertices[v2,:]

#     R = rotation(x,y,p,q)

#     # Apply the rotation
#     new_bar = (R @ barycenters.T).T

#     for j in range(12):
#         b = new_bar[j,:]

#         distances = np.linalg.norm( b - barycenters, axis = 1 )

#         flag = np.min(distances)
#         if flag >= 1E-10:
#             print('CAREFUL! Barycenters are not close')
#         match = np.argmin(distances)

#         permutations[i, j] = match


permutations = np.array(
       [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
       [ 3,  9,  7,  6, 11,  1,  0, 10,  4,  5,  2,  8],
       [ 6,  5, 10,  0,  8,  9,  3,  2, 11,  1,  7,  4],
       [ 9, 11,  7, 10,  4,  3,  6,  8,  1,  5,  0,  2],
       [ 6,  3,  0,  9,  1,  5, 10,  7,  2, 11,  8,  4],
       [10,  5,  8,  6,  2, 11,  9,  0,  4,  3,  7,  1],
       [ 2,  1,  4,  0,  7,  8,  5,  3, 11,  6, 10,  9],
       [ 5,  8, 10,  2, 11,  6,  0,  4,  9,  1,  3,  7],
       [ 0,  6,  3,  5,  9,  1,  2, 10,  7,  8,  4, 11],
       [10, 11,  9,  8,  7,  6,  5,  4,  3,  2,  0,  1],
       [ 8,  2,  4,  5,  1, 11, 10,  0,  7,  6,  9,  3],
       [ 5,  6,  0, 10,  3,  2,  8,  9,  1, 11,  4,  7],
       [ 3,  0,  6,  1,  5,  9,  7,  2, 10,  4, 11,  8],
       [ 7,  9, 11,  3, 10,  4,  1,  6,  8,  0,  2,  5],
       [ 1,  4,  2,  7,  8,  0,  3, 11,  5,  9,  6, 10],
       [11, 10,  8,  9,  5,  4,  7,  6,  2,  3,  1,  0],
       [ 9,  3,  6,  7,  0, 10, 11,  1,  5,  4,  8,  2],
       [ 7,  4,  1, 11,  2,  3,  9,  8,  0, 10,  6,  5],
       [ 1,  0,  3,  2,  6,  7,  4,  5,  9,  8, 11, 10],
       [ 2,  8,  5,  4, 10,  0,  1, 11,  6,  7,  3,  9],
       [ 4,  7, 11,  1,  9,  8,  2,  3, 10,  0,  5,  6],
       [ 8, 10,  5, 11,  6,  2,  4,  9,  0,  7,  1,  3],
       [ 4,  2,  1,  8,  0,  7, 11,  5,  3, 10,  9,  6],
       [11,  7,  9,  4,  3, 10,  8,  1,  6,  2,  5,  0],
       [ 0,  2,  5,  1,  8,  6,  3,  4, 10,  7,  9, 11],
       [ 3,  6,  9,  0, 10,  7,  1,  5, 11,  2,  4,  8],
       [ 1,  7,  4,  3, 11,  2,  0,  9,  8,  6,  5, 10],
       [11,  8,  4, 10,  2,  7,  9,  5,  1,  6,  3,  0],
       [10,  6,  5,  9,  0,  8, 11,  3,  2,  7,  4,  1],
       [ 9,  7,  3, 11,  1,  6, 10,  4,  0,  8,  5,  2],
       [ 0,  5,  6,  2, 10,  3,  1,  8,  9,  4,  7, 11],
       [ 2,  4,  8,  1, 11,  5,  0,  7, 10,  3,  6,  9],
       [ 1,  3,  7,  0,  9,  4,  2,  6, 11,  5,  8, 10],
       [ 8,  5,  2, 10,  0,  4, 11,  6,  1,  9,  7,  3],
       [11,  4,  7,  8,  1,  9, 10,  2,  3,  5,  6,  0],
       [10,  9,  6, 11,  3,  5,  8,  7,  0,  4,  2,  1],
       [ 3,  7,  1,  9,  4,  0,  6, 11,  2, 10,  5,  8],
       [ 6,  0,  5,  3,  2, 10,  9,  1,  8,  7, 11,  4],
       [ 9, 10, 11,  6,  8,  7,  3,  5,  4,  0,  1,  2],
       [ 8,  4, 11,  2,  7, 10,  5,  1,  9,  0,  6,  3],
       [ 2,  0,  1,  5,  3,  4,  8,  6,  7, 10, 11,  9],
       [ 5, 10,  6,  8,  9,  0,  2, 11,  3,  4,  1,  7],
       [ 3,  1,  0,  7,  2,  6,  9,  4,  5, 11, 10,  8],
       [ 7, 11,  4,  9,  8,  1,  3, 10,  2,  6,  0,  5],
       [ 9,  6, 10,  3,  5, 11,  7,  0,  8,  1,  4,  2],
       [ 4,  1,  7,  2,  3, 11,  8,  0,  9,  5, 10,  6],
       [ 8, 11, 10,  4,  9,  5,  2,  7,  6,  1,  0,  3],
       [ 2,  5,  0,  8,  6,  1,  4, 10,  3, 11,  7,  9],
       [ 6, 10,  9,  5, 11,  3,  0,  8,  7,  2,  1,  4],
       [ 0,  3,  1,  6,  7,  2,  5,  9,  4, 10,  8, 11],
       [ 5,  2,  8,  0,  4, 10,  6,  1, 11,  3,  9,  7],
       [ 6,  9,  3, 10,  7,  0,  5, 11,  1,  8,  2,  4],
       [10,  8, 11,  5,  4,  9,  6,  2,  7,  0,  3,  1],
       [ 5,  0,  2,  6,  1,  8, 10,  3,  4,  9, 11,  7],
       [ 4, 11,  8,  7, 10,  2,  1,  9,  5,  3,  0,  6],
       [ 7,  3,  9,  1,  6, 11,  4,  0, 10,  2,  8,  5],
       [ 1,  2,  0,  4,  5,  3,  7,  8,  6, 11,  9, 10],
       [11,  9, 10,  7,  6,  8,  4,  3,  5,  1,  2,  0],
       [ 4,  8,  2, 11,  5,  1,  7, 10,  0,  9,  3,  6],
       [ 7,  1,  3,  4,  0,  9, 11,  2,  6,  8, 10,  5]])

def which_dodecahedral_face( p , centroid ):
    '''Find which of the pentagons a point belongs to, wrt the centroid of the protein

    Params:
    p: numpy (3,1) vector of floats. The point in 3D space
    centroid: numpy (3,1) vector of floats. The centroid of the protein in 3D space


    Returns:
    Integer between 1 and 12, corresponding to the face, according to the order above

    If p == centroid, defaults to returning 1
    '''
    
    # The relative vector
    vect = p - centroid 

    # norm of the vector
    norm = np.linalg.norm(vect)

    if  norm <= 1E-10:
        # The point coincides with the centroid, default to 1
        return 1

    # Must be normalized, but the dodecahedron has "radius" sqrt(3)
    vect = np.sqrt(3) * vect / norm

    # compute distances to each barycenter (vector is automatically broadcast)
    distances = np.linalg.norm( vect - barycenters, axis = 1) 

    # find which one is minimal (if multiple, returns the smallest index)
    face = np.argmin(distances)

    # we want them from 1 to 12, not 0 to 11
    return face + 1

def dodecahedron_common_face( simplex, centroid ):
    '''Find which common face a list of vertices belongs to. Raise
    ValueError if it does not exist

    '''
    sectors = np.array([which_dodecahedral_face(p, centroid) for p in simplex], dtype = int)
    unique_sec = np.unique(sectors)

    if unique_sec.shape[0] > 1:
        raise ValueError

    else:
        return unique_sec[0]

def gen_dodecahedron_block_permutations(B):
    ''' Assume each face has a block of B descriptors. The every sample is a vector
    of 12*B entries. There are 60 elements in the rotation group of the dodecahedron

    We must return the set of these 60 permutations, keeping the elements of each 
    block fixed. When B=1, this is just the 60 permutations.
    '''

    block_permutations = np.zeros((60, 12*B), dtype=int)
    for i in range(60):

        perm = permutations[i,:]

        for j in range(12):

            target_ind = perm[j]

            block_permutations[ i , B*j:B*(j+1) ] = np.array( np.arange(B*target_ind , B*(target_ind+1) ))
        
    
    return block_permutations.tolist()

    