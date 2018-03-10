import math


class Cartographer(object):

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.top_map = {}

    # Returns the list of nodes on the curve for the given resolution.
    def get_nodes(self, resolution, two_dimensional, symmetric, input_nodes):

        if resolution not in self.top_map:
            if symmetric:
                self.top_map[resolution] = sym_z_map(resolution, two_dimensional)
            else:
                self.top_map[resolution] = z_map(resolution, two_dimensional)

        nodes = []

        if input_nodes:
            query = self.inputs
        else:
            query = self.outputs

        for node in discretize(query, resolution, two_dimensional):

            nodes.append(self.top_map[resolution][node])

        return nodes


def unpart1by1(n):
    n &= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n


def deinterleave(n):
    return unpart1by1(n >> 1), unpart1by1(n)


# special function used in converting z-ordering to
# symmetric z-ordering
def twirl(n):
    mask = 0x80000000

    for i in range(0, 15):
        n ^= (n & (mask >> (2 * i + 1))) >> 1
        n ^= (n & (mask >> (2 * i))) >> 2

    return n

# special function used in converting z-ordering to
# half symmetric z-ordering
def half_twirl(n):
    mask = 0x80000000

    for i in range(0, 15):
        # n ^= (n & (mask >> (2 * i + 1))) >> 1
        n ^= (n & (mask >> (2 * i))) >> 2

    return n


# convert cartesian coordinates to row col ordering
def discretize(coordinates, resolution, two_dimensional=True):

    nodes = []

    if two_dimensional:
        side_size = int(math.sqrt(resolution))
    else:
        side_size = int(round(resolution ** (1./3.)))

    step = 2/(side_size-1)

    if two_dimensional:
        for element in coordinates:
            x_pos, y_pos = element
            row = round((x_pos + 1) / step)
            col = round((y_pos + 1) / step)
            nodes.append(row + side_size * col)
    else:
        for element in coordinates:
            x_pos, y_pos, z_pos = element
            row = round((x_pos + 1) / step)
            col = round((y_pos + 1) / step)
            page = round((z_pos + 1) / step)
            nodes.append(row + side_size * col + side_size * side_size * page)


    return nodes


# Creates a mapping from row-col order to Z-order
def z_map(size, two_dimensional=True):
    mapping = {}

    if two_dimensional:
        side = int(math.sqrt(size))
    else:
        side = int(round(size ** (1./3.)))
    for i in range(0, size):
        if two_dimensional:
            xpos, ypos = deinterleave(i)
            # print(xpos, " ", ypos)
            mapping[ypos*side + xpos] = i
        else:
            xpos, ypos, zpos = deinterleave3d(i)
            # print(xpos, " ", ypos, " ", zpos)
            mapping[(zpos * side * side) + (ypos * side) + xpos] = i

    return mapping



# Creates a mapping from row-col order to Symmetric Z-order
def sym_z_map(size, two_dimensional=True):
    mapping = {}

    if two_dimensional:
        side = int(math.sqrt(size))
    else:
        side = int(round(size ** (1./3.)))
    for i in range(0, size):
        if two_dimensional:
            xpos, ypos = deinterleave(half_twirl(i))
            # print(xpos, " ", ypos)
            mapping[ypos*side + xpos] = i
        else:
            xpos, ypos, zpos = deinterleave3d(half_twirl3d(i))
            # print(xpos, " ", ypos, " ", zpos)
            mapping[(zpos * side * side) + (ypos * side) + xpos] = i

    return mapping

# 3d version of unpart1by1
def unpart3d(x):
    x = x & 0x9249249249249249;
    x = (x | (x >> 2))  & 0x30c30c30c30c30c3
    x = (x | (x >> 4))  & 0xf00f00f00f00f00f
    x = (x | (x >> 8))  & 0x00ff0000ff0000ff
    x = (x | (x >> 16)) & 0xffff00000000ffff
    x = (x | (x >> 32)) & 0x00000000ffffffff
    return x

# 3d version of deinterleave
def deinterleave3d(n):
    return unpart3d(n >> 2), unpart3d(n >> 1), unpart3d(n)

# only works for small sizes of n (<4097). needs fixing.
def half_twirl3d(n):

    mask = 0x800

    for i in range(0,3):
        n ^= (n & mask) >> 3
        #n ^= (n & (mask >> 1)) >> 3
        #n ^= (n & (mask >> 2)) >> 1
        #n ^= (n & (mask >> 2)) >> 2
        mask = mask >> 3

    return n

# only works for small sizes of n (<4097). needs fixing.
def twirl3d(n):

    mask = 0x800

    for i in range(0,3):
        n ^= (n & mask) >> 3
        n ^= (n & (mask >> 1)) >> 3
        n ^= (n & (mask >> 2)) >> 1
        n ^= (n & (mask >> 2)) >> 2
        mask = mask >> 3

    return n
