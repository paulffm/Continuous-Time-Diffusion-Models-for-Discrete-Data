"""
    Written by turidus (github.com/turidus) in python 3.6.0
    Dependend on Pillow 4.2, a fork of PIL (https://pillow.readthedocs.io/en/4.2.x/index.html)
"""
from PIL import Image, ImageDraw, ImageColor
import random as rnd
import re
import torchvision.transforms as transforms


class Maze:
    """This Class represents a Maze. After init it consists of an unformed maze made out of a nested list (grid) of
    untouched floor tiles. It size in X and Y are dependent on input.
    It depends on Pillow, a PIL fork (https://pillow.readthedocs.io/en/4.2.x/index.html).

    The finale internal representation of the maze is a nested list of touched floor tiles that are connected to at
    least one neighbour. Walls will be added by the graphic representation.

    The maze can be formed by two different algorithms, modified Prim's and Growing Tree. A short
    explanation of the used algorithms (and many more) can be found at http://www.astrolog.org/labyrnth/algrithm.htm
    A more in depth explanation can be found in this article series:
    Prims Algorithm: http://weblog.jamisbuck.org/2011/1/10/maze-generation-prim-s-algorithm
    Growing Tree Algorithm: http://weblog.jamisbuck.org/2011/1/27/maze-generation-growing-tree-algorithm

    This classes raises Maze Error on invalid input; when a maze get the command to change after it was formed or
    if the maze gets the command to make a graphical representation of an unformed maze.

    Content:

    private subclass    Maze Tile:  Structure representing a single tile.
    private subclass    Maze Error: Custom Error

    private function    __init_(int,int,string):    this function takes two integers for size(X,Y)
                                                    and an optional string for the name of the Maze.
    private function    __str__:    Returns a formated string with as many columns and lines as sizeX and sizeY respectivly.
                                    The cells are filled with the status of the tile (workedOn, True or False)
    private function    __repr()__ :    Returns a string with the size of the Maze as description
    private function    __getNextTiles(int,int):  returns a list of available tiles to the specified coordinates
    private function    __connectTiles(tileA,tileB): connects specified tiles to make a way
    private function    __connectTilesWithString(tile,string): connects two tiles dependend on one tile and a given connection string.
    private function    __makeEntryandExit(): creates a entry and an exit into the maze

    public function     makeMazeSimple:():  returns True
                                            This function takes the unformed maze and forms it with the modified Prim's
                                            algorithm. This results in an simple to solve maze. This algorithms is less
                                            efficient than the Growing Tree algorithm.
    public function     makeMazeGrowingTree(int,int): returns True
                                                      This algorithm forms the maze with the Growing Tree algorithmus
                                                      takes two integer values between 0 and 100, with the first
                                                      integer bigger than the second one. These are are weights defining the
                                                      behavior of the algorithm. (see link above)
    public function     makeMazeBraiding(int): This function workes as braider on a formed maze. Can either work as dead end remover (-1)
                                                or produce random loops (0-100), decided by the weight.

    public function     makePP(var,int,int,int):    Takes an optional string for color mode and two optional argument
                                                    defining the color of wall and floor.

                                                    Returns an image object.

                                                    This function takes a formed maze and creates a picture with the help of Pillow.
                                                    The size of the picture depends on the chosen pixelSizePerTiles and the amount of tiles

    public function     saveImage(image,string): Specialized implementation of Pillow's Save function. Takes an image and
                                                        saves it with an (optional) given name/path object and format.
                                                        If no name is given, a name will be constructed.
    """

    class __MazeTile:
        """This subclass is a structure representing a single tile inside the Maze. This tile has a X and Y coordinate which are specified on generation.
        It can also be specified if the tile is a wall or a floor.
        """

        def __init__(self, X, Y, isWall=True):
            """Generator for the __MazeTile class"""

            self.workedOn = False  # Needed for certain generation algorithm to define if a tile as already be touched
            self.wall = isWall  # Is the tile a wall (True) or a floor (False). Mostly for future proving
            self.coordinateX = X  # Defining the X coordinate
            self.coordinateY = Y  # Defining the Y coordinate
            self.connectTo = (
                []
            )  # A list of strings that describe the tiles this tile is connected to (North, South, West, East)

        def __str__(self):
            return (
                "X: "
                + str(self.coordinateX)
                + " "
                + "Y: "
                + str(self.coordinateY)
                + " "
                + str(self.connectTo)
            )

        def __repr__(self):
            return "__MazeTile, wall = {}, worked on = {} ,x = {}, y = {} ---".format(
                self.wall, self.workedOn, self.coordinateX, self.coordinateY
            )

    class __MazeError(Exception):
        """Custom Maze Error, containing a string describing the error that occurred
        and an errorcode:

        errorcode       meaning
        1               A wrong value was passed
        2               Out of bounds of list
        3               A maze algorithm tried to change a already changed maze
        4               A function that assumed a formed maze found an unformed maze
        """

        def __init__(self, string, errorcode):
            self.string = string
            self.errorcode = errorcode

        def __str__(self):
            return str(self.string) + " |Errorcode: {}".format(self.errorcode)

    def __init__(self, dimensionX, dimensionY, mazeName="A_Maze"):
        """Generator for the Maze class.
        It takes two integer to decide the size of the maze (X and Y). It also takes an optional string to determine the name of the maze.

        """

        if not isinstance(dimensionX, int) or not isinstance(
            dimensionY, int
        ):  # Checking input errors
            raise self.__MazeError("Maze dimensions have to be an integer > 0", 1)

        if dimensionX < 1 or dimensionY < 1:
            raise self.__MazeError("Maze dimensions have to be an integer > 0", 1)

        if not isinstance(mazeName, str):
            raise self.__MazeError("The name of the Maze has to be a string", 1)

        self.sizeX = (
            dimensionX  # The size of the Maze in the X direction (from left to right)
        )
        self.sizeY = (
            dimensionY  # The size of the Maze in the Y direction (from up to down)
        )

        self.name = mazeName  # The name of the Maze. Can be any string
        self.__mazeIsDone = False  # When this flag is False, no picture can be made. When this flag is True, the maze can not be changed

        self.mazeList = (
            []
        )  # A nested List of maze Tiles. The internal representation of the maze

        self.wallList = (
            []
        )  # A list of all lists that are walls (needed for certain algorithm)
        self.tileList = []  # A single list of all tiles (needed of certain algorithm)

        self.mazeString = ""  # A string describing the Maze in a pseudo graphical manner, gets generated everytime __str__() gets called

    def __str__(self):
        """Generates the mazeString which is a string with as many columns as sizeX and as many lines as sizeY.
        The cells are filled with the conenction of the tile at these coordinates.
        """

        self.mazeString = ""

        for row in self.mazeList:
            for tile in row:
                self.mazeString += "{:^20}".format(str(tile.connectTo))

            self.mazeString += "\n"

        return self.mazeString

    def __repr__(self):
        """Generates a representing string"""

        return "This is a Maze with width of {} and height of {}".format(
            self.sizeX, self.sizeY
        )

    def __getNextTiles(self, X, Y):
        """
        This function collects all nearest neighbour of a tile. Important for tiles that lay on a border.

        """

        if X < 0 or Y < 0:  # Checks input error (this should never happen)
            raise self.__MazeError("Inputs have to be an integer > 0", 1)

        templist = []

        try:
            if Y == 0:
                pass
            else:
                templist.append(self.mazeList[Y - 1][X])

        except IndexError:
            pass

        try:
            templist.append(self.mazeList[Y + 1][X])
        except IndexError:
            pass

        try:
            if X == 0:
                pass
            else:
                templist.append(self.mazeList[Y][X - 1])
        except IndexError:
            pass

        try:
            templist.append(self.mazeList[Y][X + 1])
        except IndexError:
            pass

        return templist

    def __connectTiles(self, tileA, tileB):
        """Takes two tiles and returns True if successful.
        Connect the two given tiles to make a way. This is used to decide where walls shouldn't be in the final picture.
        The Tile connectTo field is appended by the compass direction of the tile it connects to (N,S,E,W).
        """
        X1 = tileA.coordinateX
        Y1 = tileA.coordinateY

        X2 = tileB.coordinateX
        Y2 = tileB.coordinateY

        if X1 == X2:
            if Y1 < Y2:
                tileA.connectTo.append("S")
                tileB.connectTo.append("N")

            elif Y1 > Y2:
                tileA.connectTo.append("N")
                tileB.connectTo.append("S")

        else:
            if X1 < X2:
                tileA.connectTo.append("E")
                tileB.connectTo.append("W")

            else:
                tileA.connectTo.append("W")
                tileB.connectTo.append("E")

        return True

    def __connectTilesWithString(self, tile, direction):
        """Takes one tile and a direction string (N,S,W,E) to connect two tiles. Returns True.
        Raises __MazeError if a tile wants to connect out of the maze.
        Make sure that this only connects unconnected tiles
        """

        if direction == "N":
            try:
                if (
                    tile.coordinateY == 0
                ):  # This prevents list[-1] situations and two holes in the border wall
                    raise IndexError

                self.mazeList[tile.coordinateY - 1][tile.coordinateX].connectTo.append(
                    "S"
                )
                tile.connectTo.append("N")

            except IndexError:
                raise self.__MazeError("This tile can not connect in this direction", 2)

        elif direction == "S":
            try:
                self.mazeList[tile.coordinateY + 1][tile.coordinateX].connectTo.append(
                    "N"
                )
                tile.connectTo.append("S")

            except IndexError:
                raise self.__MazeError("This tile can not connect in this direction", 2)

        elif direction == "W":
            try:
                if tile.coordinateX == 0:
                    raise IndexError
                self.mazeList[tile.coordinateY][tile.coordinateX - 1].connectTo.append(
                    "E"
                )
                tile.connectTo.append("W")

            except IndexError:
                raise self.__MazeError("This tile can not connect in this direction", 2)

        elif direction == "E":
            try:
                self.mazeList[tile.coordinateY][tile.coordinateX + 1].connectTo.append(
                    "W"
                )
                tile.connectTo.append("E")

            except IndexError:
                raise self.__MazeError("This tile can not connect in this direction", 2)

        else:
            raise self.__MazeError("This was not a direction string", 1)

        return True

    def __makeEntryandExit(self, random=False):
        """Takes an optional boolean
        If random is set to True, it chooses the entry and exit field randomly.
        It set to False, it chooses the left upper most and right lower most corner as entry.
        """
        if random:
            tile = rnd.choice(self.mazeList[0])
            tile.connectTo.append("N")

            tile = rnd.choice(self.mazeList[-1])
            tile.connectTo.append("S")
        else:
            self.mazeList[0][0].connectTo.append("N")
            self.mazeList[-1][-1].connectTo.append("S")

        return True

    def makeMazeSimple(self):
        """Algorithm to form the final maze. It works like the modified Prim's algorithm
        (http://weblog.jamisbuck.org/2011/1/10/maze-generation-prim-s-algorithm)

        It works on the initial mazeList. The finale mazeList consist of touched floor tiles that are connect
        with each other.
        The end result is a easy to solve perfect 2D maze.

        This runs much slower than GrowTreeAlgorithm.


        A short description of what happens:
        At first a untouched (tile.workedOn = False) tile is randomly chosen, it is transformed to a
        touched tile and all neighbour are put into a frontier list.

        From this list comes the next tile which is again transformed into a touched tile
        and removed from the frontier list.
        Then this touched tile is connected to a randomly neighbor that was also touched.
        All neighbours that are untouched tiles are added into the frontier list.

        This will run until the frontier list is empty.
        """

        if self.__mazeIsDone:  # Can only run if the maze is not already formed
            raise self.__MazeError("Maze is already done", 3)

        for indexY in range(
            0, self.sizeY
        ):  # This loops generates the mazeList and populates it with new untouched floor tiles
            templist = []

            for indexX in range(0, self.sizeX):
                newTile = self.__MazeTile(indexX, indexY, isWall=False)
                templist.append(newTile)

            self.mazeList.append(templist)

        frontList = []  # A list of all untouched tiles that border a touched tile
        startingtile = rnd.choice(
            rnd.choice(self.mazeList)
        )  # A randomly chosen tile that acts as starting tile

        startingtile.workedOn = (
            True  # This flag always gets set when a tile has between worked on.
        )
        frontList += self.__getNextTiles(
            startingtile.coordinateX, startingtile.coordinateY
        )  # populates the frontier
        # list with the first 2-4 tiles

        while (
            len(frontList) > 0
        ):  # When the frontier list is empty the maze is finished because all tiles have been connected
            newFrontTiles = []
            workedOnList = []

            rnd.shuffle(frontList)
            nextTile = frontList.pop()
            nextTile.workedOn = True

            tempList = self.__getNextTiles(nextTile.coordinateX, nextTile.coordinateY)

            for (
                tile
            ) in (
                tempList
            ):  # Finds all neighbours who are touched and all that are a untouched
                if tile.workedOn:
                    workedOnList.append(tile)

                else:
                    if not tile in frontList:
                        newFrontTiles.append(tile)

            frontList += newFrontTiles

            if len(workedOnList) > 1:  # Chooses the neighbor the tile should connect to
                connectTile = rnd.choice(workedOnList)

            else:
                connectTile = workedOnList[0]

            self.__connectTiles(nextTile, connectTile)

        self.__makeEntryandExit()  # Finally produces a Entry and an Exit
        self.__mazeIsDone = True
        return True

    def makeMazeGrowTree(self, weightHigh=99, weightLow=97):
        """Algorithm to form the final maze. It works like the Grow Tree algorithm
        http://weblog.jamisbuck.org/2011/1/27/maze-generation-growing-tree-algorithm

        It works on the initial mazeList. The finale mazeList consist of touched floor tiles that are connect
        with each other.
        The end result is a perfect 2D maze that has a variable hardness in solution

        This runs much faster than makeMazeSimple and should be used as default.

        This algorithm can be modified with two weights between 0 and 100.
        weightHigh should always be higher or equal to weightLow.

        Three extrems:

        weightHigh == 100, weightLow == 100:

            The algorithm always takes the the newst tile out of the choice list and behaves
            like a reversed backtrace. Very hard to solve
            http://weblog.jamisbuck.org/2010/12/27/maze-generation-recursive-backtracking

        weightHigh == 100, weightLow == 0

            The algorithm always chooses the next tile randomly
            and behaves like Prim's algorithm, but runs much faster.
            Solving is less hard than 100/100


        weightHigh == weightLow == 0:

            The algorithm always chooses the oldest tile in the list
            The resulting maze will have a lot of long passageways (high river factor) and
            will be trivial to solve.


        I personally like a 99/97 distribution but feel free to experiment yourself.

        A short description of what happens:
        At first one untouched tile is randomly chosen, it is transformed to a
        touched tile and it is put into a list of available tiles.

        From this list is a tile chosen, how depends on the weights.
        If this tile has no untouched neighbours it is removed from the list of available tiles.
        Else a neighbour is chosen, marked as touched, put into the availabe tile list and
        connected to the current tile.
        This loops until the list of available tiles is empty.
        """

        if (
            self.__mazeIsDone
        ):  # This function only runs of the Maze is not already formed.
            raise self.__MazeError("Maze is already done", 3)

        for indexY in range(
            0, self.sizeY
        ):  # This loops generates the mazeList and populates it with new untouched floor tiles
            templist = []

            for indexX in range(0, self.sizeX):
                newTile = self.__MazeTile(indexX, indexY, isWall=False)
                templist.append(newTile)

            self.mazeList.append(templist)

        startingtile = rnd.choice(
            rnd.choice(self.mazeList)
        )  # First tile is randomly chosen
        startingtile.workedOn = True

        choiceList = [startingtile]  # The list of available tiles

        while len(choiceList) > 0:  # Runs until choiceList is empty
            choice_ = (
                rnd.random() * 100
            )  # This random choice determines how the next tile is chosen

            if choice_ <= weightLow:
                nextTile = choiceList[-1]
            elif weightLow < choice_ < weightHigh:
                nextTile = rnd.choice(choiceList)
            else:
                nextTile = choiceList[0]

            neiList = []  # List of neighbours

            for tile in self.__getNextTiles(nextTile.coordinateX, nextTile.coordinateY):
                if not tile.workedOn:
                    neiList.append(tile)

            if (
                len(neiList) == 0
            ):  # either removing this tile or choosing a neighbour to interact with
                choiceList.remove(nextTile)

            else:
                connectTile = rnd.choice(neiList)
                connectTile.workedOn = True
                choiceList.append(connectTile)
                self.__connectTiles(nextTile, connectTile)

        self.__makeEntryandExit()  # finally marking an Entry and an Exit
        self.__mazeIsDone = True
        return True

    def makeMazeBraided(self, weightBraid=-1):
        """This function produces a braided maze by either removing dead ends or by producing
        random loops. It takes an Interger betwee -1 and 100.

        weightBraid decides how many percent of the tiles should creat loops. (0-100)
        If set to -1, it will braid the maze by removing all and only dead ends

        Raises a __MazeError on wrong input or if the maze is unformed
        """
        if not self.__mazeIsDone:
            raise self.__MazeError("Maze needs to be formed first", 4)

        if not isinstance(weightBraid, int) or weightBraid < -1 or weightBraid > 100:
            raise self.__MazeError("weightBraid has to be >= -1", 1)

        elif weightBraid == -1:
            for row in self.mazeList:
                for tile in row:
                    if (
                        len(tile.connectTo) == 1
                    ):  # All tiles that are only connected to one other tile are dead ends
                        directionList = ["N", "S", "W", "E"]
                        directionList.remove(tile.connectTo[0])

                        rnd.shuffle(directionList)  # Randomizing connections
                        for direction in directionList:
                            try:
                                self.__connectTilesWithString(tile, direction)
                                break

                            except self.__MazeError as mazeExcept:
                                if mazeExcept.errorcode == 2:
                                    pass

                                else:
                                    raise
        else:
            for row in self.mazeList:
                for tile in row:
                    if weightBraid >= (
                        rnd.random() * 100
                    ):  # Weight decides if this tiles gets a addtional connection
                        directionList = ["N", "S", "W", "E"]
                        for connection in tile.connectTo:
                            directionList.remove(connection)

                        rnd.shuffle(directionList)  # Randomizing connections
                        for direction in directionList:
                            try:
                                self.__connectTilesWithString(tile, direction)
                                break

                            except self.__MazeError as mazeExcept:
                                if mazeExcept.errorcode == 2:
                                    pass

                                else:
                                    raise

        return True

    def makePP(self, mode="1", colorWall=0, colorFloor=1, pixelSizeOfTile=2):
        """
        This generates and returns a Pillow Image object. It takes into account the size of the maze and
            the size of the the indivual pixel defined with pixelSizeOfTile. Defaults to 10 pixel.

            It create this picture by drawing a square with the defined size for every tile in
            the mazeList on a background. It then proceeds to draw in the connections this tile has by checking
            tile,connectTo.

            The default mode this picture is created is 1 bit per pixel and allows only for white (1) and black(0)
            pictures.

            The mode and colors can be changed to RGB, but this will increase the picture size massivly (24 times).

            Allowed modes:
            "1":    1 bit per pixel, colors are 1-bit intergers. 1 for white and 0 for black
            "RGB":  3x8 bit per pixels. colors can be given either as three 8 bit tuples (0,0,0)-(255,255,255)
                    or html color strings.

            The pixelSizeOfTile decides the edge length on of tile square in the picture.

            Raises __MazeError if the maze is not already finished and on wrong input.
        """
        if not self.__mazeIsDone:
            raise self.__MazeError("There is no Maze yet", 4)

        if mode == "1":  # Checking for input errors
            if colorWall in (1, 0) and colorFloor in (1, 0):
                pass
            else:
                raise self.__MazeError(
                    "In mode '1' the color vaules have to be 0 for black or 1 for white",
                    1,
                )

        elif mode == "RGB":
            try:
                if isinstance(colorWall, str):
                    colorWall = ImageColor.getrgb(colorWall)

                elif isinstance(colorWall, tuple) and len(colorWall) == 3:
                    for i in colorWall:
                        if not isinstance(i, int) or (i < 0 or i > 255):
                            raise self.__MazeError(
                                "RGB mode excepts only 8-bit integers", 1
                            )

                else:
                    raise self.__MazeError(
                        "RGB Mode only excepts color strings or 3x8bit tulpels", 1
                    )

                if isinstance(colorFloor, str):
                    colorFloor = ImageColor.getrgb(colorFloor)

                elif isinstance(colorFloor, tuple) and len(colorFloor) == 3:
                    for i in colorFloor:
                        if not isinstance(i, int) or (i < 0 or i > 255):
                            raise self.__MazeError(
                                "RGB mode excepts only 8-bit integers", 1
                            )

                else:
                    raise self.__MazeError(
                        "RGB Mode only excepts color strings or 3x8bit tulpels", 1
                    )

            except ValueError:
                raise self.__MazeError(
                    "RGB mode excepts 140 common html color strings. This was not one of them",
                    1,
                )

        else:
            raise self.__MazeError(
                "The mode was not recognized. Only '1' or 'RGB' are allowed", 1
            )

        if not isinstance(pixelSizeOfTile, int) or pixelSizeOfTile <= 0:
            raise self.__MazeError(
                "the size of the tiles has to be an integer > 0", 1
            )  # Finished looking for input errors.

        size = (
            pixelSizeOfTile * (self.sizeX * 2 +1),
            pixelSizeOfTile * (self.sizeY * 2 +1),
        )
        # Determines the size of the picture. It does this by taking the number of tiles,
        # multiplying it with 2 to account for walls or connections and adds one for offset

        image = Image.new(mode, size, colorWall)  # Generates a Pillow Image object
        drawImage = ImageDraw.Draw(image)

        for row in self.mazeList:  # Iterates over all tiles
            for tile in row:
                # There are floor tiles at postion 1,3,5..., at postion 0,2,4,6... are either wall tiles or connecting tiles.

                x = ((tile.coordinateX + 1) * 2 -1) * pixelSizeOfTile
                y = ((tile.coordinateY + 1) * 2 -1) * pixelSizeOfTile
                drawImage.rectangle(
                    [x, y, x + pixelSizeOfTile - 1, y + pixelSizeOfTile - 1],
                    fill=colorFloor,
                )

                if "N" in tile.connectTo:
                    drawImage.rectangle(
                        [x, y - pixelSizeOfTile, x + pixelSizeOfTile - 1, y - 1],
                        fill=colorFloor,
                    )

                if "S" in tile.connectTo:
                    drawImage.rectangle(
                        [
                            x,
                            y + pixelSizeOfTile,
                            x + pixelSizeOfTile - 1,
                            y + pixelSizeOfTile + pixelSizeOfTile - 1,
                        ],
                        fill=colorFloor,
                    )

                if "W" in tile.connectTo:
                    drawImage.rectangle(
                        [x - pixelSizeOfTile, y, x - 1, y + pixelSizeOfTile - 1],
                        fill=colorFloor,
                    )

                if "E" in tile.connectTo:
                    drawImage.rectangle(
                        [
                            x + pixelSizeOfTile,
                            y,
                            x + pixelSizeOfTile + pixelSizeOfTile - 1,
                            y + pixelSizeOfTile - 1,
                        ],
                        fill=colorFloor,
                    )

        return image  # .resize((28, 28))

    def saveImage(self, image, n, name=None, format=None, pixelSizeOfTile=1):
        """Specialized implementation of Pillow's Save function. Takes an image and
        saves it with an (optional) given name or format. Either this name has contains a
        file extension that is known to Pillow or format has to be specified.

        This name has to fullfile all filename rules on the given system!

        The name can be a path object, then the format argument needs to be specified.

        Details here: https://pillow.readthedocs.io/en/4.2.x/reference/Image.html#PIL.Image.Image.save

        If no name is given, a name will be constructed and a png file created.
        The name is constructed out of the maze name and its pixel size in x and y direction.
        To make sure that the image can be saved, all chars that are not letters, numbers or underscores
        will be removed from the maze name and the length will be limited two 120 chars.
        This will not be done on names that are passed as arguments!
        """
        if name == None:
            tempName = re.sub(
                r"[^a-zA-Z0-9_]", "", self.name
            )  # Regular expression to make name filename safe
            if len(tempName) > 120:  # Limiting the length of the filename
                tempName = tempName[0:120]
            size = (
                pixelSizeOfTile * (self.sizeX * 2 + 1),
                pixelSizeOfTile * (self.sizeY * 2 + 1),
            )
            name = tempName + "-" + str(size[0]) + "_" + str(size[1]) + ".png"

        image_path = f"/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/TAUnSDDM/lib/datasets/rectangular_mazes/train/{n}.png"
        image.save(image_path)
        return True


import torchvision.transforms as transforms

def maze_gen(
    limit: int,
    size: int=None,
    crop: bool=False,
    dim_x: int = 10,
    dim_y: int = 10,
    pixelSizeOfTile: int = 1,
    weightHigh: int = 99,
    weightLow: int = 99,
    device = 'cpu'
):
    n = 1
    # Fügen Sie hier die Resize-Transformation hinzu
    transform = transforms.Compose([
        transforms.PILToTensor() # Ändern Sie dies auf (32, 32) für 32x32 Größe
    ])
    image_list = []
    while True:
        newMaze = Maze(dim_x, dim_y, mazeName=f"maze_{n}")
        newMaze.makeMazeGrowTree(weightHigh, weightLow)
        mazeImageBW = newMaze.makePP(pixelSizeOfTile=pixelSizeOfTile)
        if crop:
            cropped_size = pixelSizeOfTile * (dim_x * 2 + 1) - 1
            mazeImageBW = mazeImageBW.crop((1, 1, cropped_size, cropped_size))
        
        # Wenden Sie die Transformation an und resizen Sie das Bild
        img_tensor = transform(mazeImageBW)
        
        # Verschieben Sie das Tensor auf das gewünschte Gerät
        img_tensor = img_tensor.to(device)
        
        image_list.append(img_tensor)
        #newMaze.saveImage(mazeImageBW, n)
        if n == limit:
            break
        n += 1

    return image_list



# Examples:
# newMaze = Maze(10,10)
# newMaze.makeMazeGrowTree(weightHigh = 99, weightLow = 97)
# mazeImageBW = newMaze.makePP()
# mazeImageBW.show() #can or can not work, see Pillow documentation. For debuging only
# newMaze.saveImage(mazeImageBW)

# newMaze = Maze(50,50,mazeName="ColorMaze")
# newMaze.makeMazeGrowTree(90.80)
# mazeImageColor = newMaze.makePP(mode="RGB",colorWall = "yellow",colorFloor = (250,0,20))
# newMaze.saveImage(mazeImageColor)

# newMaze = Maze(30,30,mazeName = "BraidedMaze")
# newMaze.makeMazeGrowTree(weightHigh = 100, weightLow = 95)
# newMaze.makeMazeBraided(-1)
# mazeImageBW.show()
# newMaze.saveImage(mazeImageBW)
