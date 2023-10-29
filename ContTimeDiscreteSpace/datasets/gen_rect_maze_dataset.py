'''
Dataset that creates Rectangular Mazes
'''
import math
import time
from sys import argv

from hashlib import sha256
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
import math
import time
import pprint
import random
from PIL import Image, ImageDraw
from typing import List, Any

'''
Disjoint Set data structure
Also known as Union Find
'''

class DisjointSet:
    def __init__(self, items: List[Any]):
        # 1. assign a mapping to items
        # since it is a list, let's assume each item is mapped to its index
        self.items = items

        # make every item its own parent
        self.parent = [idx for idx in range(len(items))]

        # size of each set/tree is one at the start (set containing the item itself).
        self.size = [1 for _ in range(len(items))]

    def find(self, x: int) -> int:
        # find root node for give node x
        item = x
        while item != self.parent[item]:
            item = self.parent[item]
        root = item

        # path compression (optional): makes subsequent finds faster:
        # point all the nodes from x till root directly to root
        while x != root:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent

        return root

    def union(self, a: int, b: int) -> int:
        # takes two nodes and combines their parent into a single tree

        # we keep track of items in sets using indices. so get indices of provided items
        # a = self.items.index(item_a)
        # b = self.items.index(item_b)

        # find parents of both items
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            # add the smaller set to the larger one
            # this keeps the depth of larger tree minimum
            if self.size[root_a] < self.size[root_b]:
                self.size[root_b] += self.size[root_a]
                self.parent[root_a] = root_b
            else:
                self.size[root_a] += self.size[root_b]
                self.parent[root_b] = root_a

        # both a & b are part of the same set (if they weren't already)
        # and the root of both is root_a
        return root_a

'''
Creates Dataset of Rectangular Mazes
(actually there are Squared)
'''
class KruskalRectangular:
    '''
    Creates a Spanning Tree for given graph while picking edges at random and including their vertices in the graph
    if not already in such a way that there are no cycles

    This implementation is specific to Squared Maze from a Squared Grid Graph (nxn).
    '''

    def __init__(self, n):

        self.n = n
        self.total_cells = n ** 2

        self.TOP = 0
        self.RIGHT = 1
        self.BOTTOM = 2
        self.LEFT = 3

    def create_graph(self):
        # creates a graph aligned with squared grid pattern

        # we do not need adjacency list for Kruskal's, only the list of edges
        # but code is there for adjacency list for Grid graph, if we need it in the future
        # graph = {
        #     cell: [0, 0, 0, 0] for cell in range(self.n ** 2)
        # }

        edges = []

        # create a Grid graph where every node/cell is connected in all 4 directions
        # except the boundary cells
        for row in range(self.n):
            for col in range(self.n):
                # convert to 1D index
                node = self.n * row + col

                # only add connection to the right and bottom cells for each cell
                # this avoid duplication

                if col < self.n - 1:
                    # graph[node][self.RIGHT] = 1

                    right_node = node + 1
                    # write the smaller node first
                    edges.append((node, right_node))

                if row < self.n - 1:
                    # graph[node][self.BOTTOM] = 1
                    bottom_node = node + self.n
                    edges.append((node, bottom_node))

        # return graph, edges
        return edges

    def kruskal_spanning_tree(self):
        # creates s Spanning Tree using Randomized Kruskal's for given graph while picking edges at random and
        # including their vertices in the graph if not already in such a way that there are no cycles

        edges_of_graph = self.create_graph()

        # the minimum spanning tree has no edges in the start
        # PARENT, LEFT, RIGHT, CHILD
        spanning_tree = {
            cell: [0, 0, 0, 0] for cell in range(self.total_cells)
        }

        # record all the edges involved in Spanning Tree
        edges = []

        # cell indices will be used in disjoint set and then to map back to real edge
        cells = [idx for idx in range(self.total_cells)]

        disjoint = DisjointSet(cells)

        random.shuffle(edges_of_graph)

        for edge in edges_of_graph:
            # pick one edge at random, connecting to a new cell that is not already in visited.
            cell1, cell2 = edge

            # if these cells are not part of the same set, then they are separate trees and we need to combine them
            if disjoint.find(cell1) != disjoint.find(cell2):
                disjoint.union(cell1, cell2)

                # connect these two nodes in the spanning tree
                direction = self.get_neighbour_dir(cell1, cell2)
                spanning_tree[cell1][direction] = 1

                # also set it vice versa
                neighbour_dir = self.get_neighbour_dir(cell2, cell1)
                spanning_tree[cell2][neighbour_dir] = 1

                # also add to the list of edges for our spanning tree
                edges.append(edge)

        return spanning_tree, edges

    def index_2d(self, cell_1d):

        row = cell_1d // self.n
        col = cell_1d % self.n

        return row, col

    def get_neighbour_dir(self, cell1, cell2):
        '''
        returns the direction in which next_node lies relative to node.
        this method does not check the integrity of indices and whether this graph
        actually represents a grid pattern.
        '''
        if cell1 == cell2 + self.n:
            return self.TOP
        elif cell1 == cell2 - 1:
            return self.RIGHT
        elif cell1 == cell2 + 1:
            return self.LEFT
        elif cell1 == cell2 - self.n:
            return self.BOTTOM

    @staticmethod
    def test():
        # number of nodes in a row
        n = 4
        kr = KruskalRectangular(n)

        graph, edges = kr.create_graph()

        print("Kruskal Spanning Tree (as adjacency list):")
        pprint.pp(
            kr.kruskal_spanning_tree(edges)
        )


class RectangularKruskalMaze:

    def __init__(self, n, side_len):
        self.n = n
        self.side_len = side_len

    def create_maze_image(self, file_name=None):

        # create a spanning using Kruskal's Randomized to depict our maze
        algo = KruskalRectangular(self.n)
        spanning_tree, edges = algo.kruskal_spanning_tree()

        '''
        the indices of grid are 0, 1, 2, ...., 2*n
        between them, the cells that correspond to cells of maze (adjacency matrix) 
        have indices: 1, 3, 5, ... 2*n - 1

        in between these cells, we will insert 'filler' cells that will
        represent the connection / edge between these cells or the walls.
        that's why those gaps (even cells) exist
        '''

        grid_len = 2 * self.n + 1

        image_width = grid_len * self.side_len
        image_height = image_width
        # print('Image Dimensions:', image_width, 'x', image_height)

        maze_image = Image.new(mode='1', size=(image_width, image_height), color=(0))
        canvas = ImageDraw.Draw(maze_image)

        # center of origin for PIL is upper left corner

        # length of the side of a box / square / cell. for easier access
        side = self.side_len

        # the image is black. just draw white boxes where the cells appear
        # or where a connection between two connected cell appears.
        # we only need to draw the right and bottom connection for the current cell.
        for row in range(self.n):
            for col in range(self.n):

                x = (2 * col + 1) * side
                y = (2 * row + 1) * side

                # draw this cell
                canvas.rectangle([(x, y), (x+side, y+side)], width=side, fill='white')

                # cell index in 1D form
                node = row * self.n + col

                # draw connections too (each cell checks right & bottom connections)

                # if the node has a right neighbour and is connected to it
                if col + 1 < self.n and spanning_tree[node][algo.RIGHT] == 1:
                    canvas.rectangle([(x+side, y), (x + 2*side, y + side)], width=side, fill='white')

                # if this cell has a bottom neighbour (i.e. this cell is not in the last row)
                # and is connected to the bottom neighbour
                if row + 1 < self.n and spanning_tree[node][algo.BOTTOM] == 1:
                    canvas.rectangle([(x, y+1), (x + side, y + 2*side)], width=side, fill='white')

        # the entrance to the maze is the box in our grid at 0,side
        canvas.rectangle([(0, side), (side, 2*side)], width=side, fill='white')

        # the exit to the maze is the box in last column and 2nd last row
        x = 2 * self.n * side
        y = (2 * self.n - 1) * side
        canvas.rectangle([(x, y), (x + side, y + side)], width=side, fill='white')

        if file_name is not None:
            maze_image.save(file_name)
        else:
            # if path not provided, return edges and imaeg

            # sort the edges so they could be saved as graph representation and compared to avoid duplicate trees
            # each edge has a smaller indexed node at index 0. we will sort by using that vertex, all the edge tuples
            # and this convention will be used to detect duplicates.
            edges.sort(key=lambda edg: edg[0])

            # print("Edges in the Graph:")
            # pprint.pp(edges)
            return edges, maze_image



class RectangularDataset:

    def __init__(self, row_len, side_len, num_mazes):
        self.n = row_len
        self.side_len = side_len
        self.num_mazes = num_mazes

    def create_dataset(self):
        start_time = time.time()

        seconds = math.floor(time.time())
        cwd_path = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/datasets/'
        ds_name = f'rectangular_mazes'

        # create a folder for this dataset in current directory
        dataset_directory = cwd_path + ds_name
        Path(dataset_directory).mkdir(exist_ok=True)

        # add logs.txt
        log_file_name = f'{dataset_directory}/logs.txt'
        log_file = open(log_file_name, 'a')

        # add spanning_trees.txt to record a spanning tree per line for every maze
        tree_file_name = f'{dataset_directory}/spanning_tree.txt'
        tree_file = open(tree_file_name, 'a')

        # a map whose key is a hash of a Maze's spanning tree and value is its maze_id
        # helpful in checking duplicates
        maze_hashes = {}

        # 0 <-> n-1
        maze_id = 0

        mazes_remaining = self.num_mazes

        generator = RectangularKruskalMaze(self.n, self.side_len)

        while mazes_remaining > 0:
            edges, maze_image = generator.create_maze_image()

            spanning_tree_str = str(edges)

            # check if a maze with similar structure was already created
            # edges follow a convention and are sorted. create a hash and use hash checking
            # for faster checking for duplicates
            key = sha256(spanning_tree_str.encode()).hexdigest()
            value = maze_id

            if key in maze_hashes:
                output_line = f"Duplicate found for Maze#{maze_id} with Maze#{maze_hashes[key]}\n"
                output_line += f"{spanning_tree_str}\n"
                output_line += "Generating again\n"
                log_file.write(output_line)

                print(output_line)
                print()
                continue

            # else this is a new maze

            # maze will be saved at this path
            maze_image_path = f'{dataset_directory}/{maze_id}.png'
            maze_image.save(maze_image_path)

            output_line = f'Maze created #{maze_id}, {key}\n'
            log_file.write(output_line)
            print(output_line, end='')

            maze_hashes[key] = maze_id
            tree_file.write(f'{maze_id}: {spanning_tree_str}\n')

            maze_id += 1
            mazes_remaining -= 1

        output_line = f'\nTime Taken: {time.time() - start_time} seconds'
        log_file.write(output_line)
        print(output_line)

        log_file.close()
        tree_file.close()

def create_maze_data(num_cells_in_row, side_length, num_items):
    ds = RectangularDataset(num_cells_in_row, side_length, num_items)
    ds.create_dataset()

if __name__ == '__main__':
    # number of cells in a single row
    num_cells_in_row = 10
    side_length = 10

    # number of mazes to generate in the dataset
    num_items = 10

    ds = RectangularDataset(num_cells_in_row, side_length, num_items)

    ds.create_dataset()

    print('Task finished.')

    # if len(argv) == 4:
    #     num_cells_in_row = int(argv[1])
    #     side_length = int(argv[2])
    #     num_items = int(argv[3])
    #     ds = RectangularDataset(num_cells_in_row, side_length, num_items)
    #
    #     ds.create_dataset()
    #
    #     print('Task finished.')
    #
    # else:
    #     print('Usage:')
    #     print('python generate_rect_dataset.py NUM_CELLS CELL_SIDE_LEN NUM_MAZES')