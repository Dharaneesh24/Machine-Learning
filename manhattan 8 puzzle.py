from copy import deepcopy

# final orientation
final_matrix = []
print("Final matrix")
for i in range(3):
    final_matrix.append(list(map(int, input().split())))

mapping = dict()
for i in range(3):
    for j in range(3):
        mapping[final_matrix[i][j]] = i*3 + j


#define node
class Node:
    def __init__(self, matrix: list(list()), g: int, pos: tuple, parent) -> None:
        self.tiles = matrix
        self.g = g
        self.pos = pos
        self.h = self.find_h()
        self.f = self.g + self.h
        self.up, self.down, self.left, self.right = None, None, None, None
        self.parent = parent


    def find_h(self):
        man_dist = 0
        for i in range(3):
            for j in range(3):
                f_row = mapping[self.tiles[i][j]] // 3
                f_col = mapping[self.tiles[i][j]] % 3
                man_dist += abs(i-f_row)
                man_dist += abs(j-f_col)
        return man_dist


def order_insert(list_1: list(), node: Node): # 1 3 5 7 , 9
    size = len(list_1)
    i = 0
    while i < size and list_1[i].f <= node.f:
        i += 1
    if i == size:
        list_1.append(node)
    else:
        list_1.insert(i,node)


unique = dict()

# check for only unique states
def check(node: Node):
    if node.h not in unique:
        return True

    for n in unique[node.h]:
        next = False
        for i in range(len(n.tiles)):
            for j in range(len(n.tiles[0])):
                if n.tiles[i][j] != node.tiles[i][j]:
                    next = True
                    break
            if next:
                break
        if not next:
            return False
    return True


def is_solvable(matrix) -> bool:
    l = [] 
    inversions = 0
    for row in matrix:
        l += row
    for i in range(len(l)):
        if l[i] == 0:
            continue
        for j in range(i+1,len(l)):
            if l[j] != 0 and mapping[l[j]] < mapping[l[i]]:
                inversions += 1
    if inversions % 2 == 0:
        return True
    return False


# find children for a node
def explore(node: Node):
    row, column = node.pos[0], node.pos[1]
    children = []
    if row != 0:
        new_mat = deepcopy(node.tiles)
        new_mat[row-1][column], new_mat[row][column] = new_mat[row][column], new_mat[row-1][column] # swap blank value and its upper tile
        node.up = Node(new_mat, node.g+1, (row-1, column), node)
        children.append(node.up)
    
    if row != 2:
        new_mat = deepcopy(node.tiles)
        new_mat[row+1][column], new_mat[row][column] = new_mat[row][column], new_mat[row+1][column] # swap blank value and its lower tile
        node.down = Node(new_mat, node.g+1, (row+1, column), node)
        children.append(node.down)
             
    
    if column != 0:
        new_mat = deepcopy(node.tiles)
        new_mat[row][column-1], new_mat[row][column] = new_mat[row][column], new_mat[row][column-1] # swap blank value and its left tile
        node.left = Node(new_mat, node.g+1, (row, column-1), node)
        children.append(node.left)

    if column != 2:
        new_mat = deepcopy(node.tiles)
        new_mat[row][column+1], new_mat[row][column] = new_mat[row][column], new_mat[row][column+1] # swap blank value and its right tile
        node.right = Node(new_mat, node.g+1, (row, column+1), node)
        children.append(node.right)
    
    return children


#set final state
final = Node(final_matrix, -1, (2,2), None)

#set initial state by getting input from user
init_mat = []
print("Enter the intial state of the puzzle as a matrix")
for i in range(3):
    init_mat.append(list(map(int, input().split())))
    if 0 in init_mat[i]:
        pos = (i, init_mat[i].index(0))

initial = Node(init_mat, 0, pos, None)
if not is_solvable(init_mat):
    print("No solution exists")
else:
    open_list = [initial] #initializing the open list
    current = open_list.pop(0)
    solution = True
    g = 0
    while current.h != 0:
        if current.g > g:
            g = current.g
            print(g)

        children = explore(current)
        for child in children:
                order_insert(open_list,child)
        if len(open_list) == 0:
            print("No solution exists")
            solution = False
            break
        if current.h not in unique:
            unique[current.h] = [current]
        else:
            unique[current.h].append(current)
        current = open_list.pop(0)
        while not check(current):
            current = open_list.pop(0)
        
    if solution == True:
        path = []
        while current:
            path.insert(0,[current])
            if current.parent == None:
                path[0].append("START")
            else:
                if current.parent.up == current:
                    path[0].append("UP")
                elif current.parent.down == current:
                    path[0].append("DOWN")
                elif current.parent.left == current:
                    path[0].append("LEFT")
                elif current.parent.right == current:
                    path[0].append("RIGHT")
            current = current.parent
        for n in path:
            print(n[1], "|| g =", n[0].g, "h =", n[0].h, "f =", n[0].f)
            print(*(n[0].tiles), sep='\n')
            print()