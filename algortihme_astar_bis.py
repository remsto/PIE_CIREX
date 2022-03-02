def A_star(org, des, maze):
    open_list = []
    close_list = []
    open_list.append(org)
    while len(open_list) > 0:

        temp_node = open_list[0]
        for node in open_list:
            if node.f < temp_node.f:
                temp_node = node
        current_node = temp_node

        open_list.remove(current_node)
        close_list.append(current_node)
        # Find all the leading nodes of the current node
        node_list = []
        if Is_valid(current_node.x, current_node.y - 1, maze, open_list, close_list):
            node_list.append(Node(current_node.x, current_node.y - 1))
        if Is_valid(current_node.x, current_node.y + 1, maze, open_list, close_list):
            node_list.append(Node(current_node.x, current_node.y + 1))
        if Is_valid(current_node.x - 1, current_node.y, maze, open_list, close_list):
            node_list.append(Node(current_node.x - 1, current_node.y))
        if Is_valid(current_node.x + 1, current_node.y, maze, open_list, close_list):
            node_list.append(Node(current_node.x + 1, current_node.y))
        neighbors = node_list
        for node in neighbors:
            if node not in open_list:
                # If the current node is not open_list, marked as the parent node, and placed in open_ In the list
                node.init_node(current_node, des)
                open_list.append(node)
            # If the destination is open_list, directly return to the end grid
            for node in open_list:
                if (node.x == des.x) and (node.y == des.y):
                    return node
        # Traversing open_list, still unable to find the destination, indicating that the destination has not been reached, return empty
    return None


def Is_valid(x, y, maze, open_list=[], close_list=[]):

    # Judge whether it is out of bounds
    if x < 0 or x >= len(maze) or y < 0 or y >= len(maze[0]):
        return False
    # Judge whether there are obstacles
    if maze[x][y] == 1:
        return False
    # Is it already open_ In the list
    if Is_contain(open_list, x, y):
        return False
    # Is it already in close_ In the list
    if Is_contain(close_list, x, y):
        return False
    return True


def Is_contain(nodes, x, y):
    for node in nodes:
        if (node.x == x) and (node.y == y):
            return True
    return False


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.parent = None

    def init_node(self, parent, end):
        self.parent = parent
        if parent is not None:
            self.g = parent.g + 1
        else:
            self.g = 1
        self.h = abs(self.x - end.x) + abs(self.y - end.y)
        self.f = self.g + self.h


maze = [[0 for _ in range(1112//5)] for _ in range(750//5)]
# for i in range(200):
#     maze[119][i] = 1

# Set the start and end points
org = Node(1, 1)
des = Node(150, 50)
# Search the end of maze
result_node = A_star(org, des, maze)
# Backtracking maze path
path = []
while result_node is not None:
    path.append(Node(result_node.x, result_node.y))
    result_node = result_node.parent
# Output maze and path, the path is indicated by the * sign
cnt = 0
for i in range(0, len(maze)):
    for j in range(0, len(maze[0])):
        if Is_contain(path, i, j):
            cnt += 1
            print("*, ", end='')
        else:
            print(str(maze[i][j]) + ", ", end='')
    print()
print("The shortest path is:", cnt-1)
