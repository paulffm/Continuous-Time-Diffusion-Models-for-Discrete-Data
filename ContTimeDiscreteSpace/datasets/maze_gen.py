#ToDo: load dataset 

from maze import Maze

def main():
    n = 1
    limit = 1000
    while True:

        newMaze = Maze(10, 10, mazeName=f"maze_{n}")
        newMaze.makeMazeGrowTree()
        mazeImageBW = newMaze.makePP()

        newMaze.saveImage(mazeImageBW, n=n)

        if n == limit:
            break 
        n +=1


if __name__ == "__main__":
    main()
