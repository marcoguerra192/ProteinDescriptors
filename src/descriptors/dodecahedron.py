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


def which_dodecahedral_sector( p , centroid ):
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

def dodecahedron_common_sector( simplex, centroid ):
    '''Find which common face a list of vertices belongs to. Raise
    ValueError if it does not exist

    '''
    sectors = np.array([which_dodecahedral_sector(p, centroid) for p in simplex], dtype = int)
    unique_sec = np.unique(sectors)

    if unique_sec.shape[0] > 1:
        raise ValueError

    else:
        return unique_sec[0]

    