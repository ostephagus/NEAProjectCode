using System.IO;

namespace UserInterface.HelperClasses
{
    public class ObstacleHolder
    {
        private string? fileName;
        private bool usingObstacleFile;

        private int dataWidth;
        private int dataHeight;

        public string? FileName { get => fileName; set => fileName = value; }
        public bool UsingObstacleFile { get => usingObstacleFile; set => usingObstacleFile = value; }
        public int DataWidth { get => dataWidth; set => dataWidth = value; }
        public int DataHeight { get => dataHeight; set => dataHeight = value; }

        public ObstacleHolder(string? fileName, bool usingObstacleFile)
        {
            this.fileName = fileName;
            this.usingObstacleFile = usingObstacleFile;
            dataWidth = 128;
            dataHeight = 256;
        }

        /// <summary>
        /// Reads in an obstacle file, sets the data width and height, and returns the contents, formatted into a boolean array.
        /// </summary>
        /// <returns>The contents of the file, formatted into a flattened boolean array.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the file is not found or, more likely, the file cannot be accessed with current permissions.</exception>
        /// <exception cref="FileFormatException">Thrown when the file is not in the corrent format.</exception>
        public bool[] ReadObstacleFile()
        {
            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException("Specified file was not found. Check the correct permissions exist to access it, and that it has not been deleted.");
            }

            byte[] obstacleData;
            using (Stream stream = File.OpenRead(fileName))
            {
                using BinaryReader reader = new BinaryReader(stream);
                try
                {
                    dataWidth = reader.ReadInt32(); // Read iMax and jMax
                    dataHeight = reader.ReadInt32();
                    int fieldLength = (dataWidth + 2) * (dataHeight + 2);
                    int obstacleDataLength = fieldLength / 8 + (fieldLength % 8 == 0 ? 0 : 1); // Field length divided by 8 plus an extra byte for any remaining bits
                    obstacleData = reader.ReadBytes(obstacleDataLength);
                }
                catch (IOException e)
                {
                    string exceptionMessage = e.Message.Length > 0 ? e.Message : "[no internal error message]";
                    throw new FileFormatException($"Input file was malformed, and an error occurred while parsing: {exceptionMessage}.");
                }
            }

            bool[] obstacles = new bool[(dataWidth + 2) * (dataHeight + 2)];
            int byteNumber = 0;
            for (int i = 0; i < obstacles.Length; i++)
            {
                obstacles[byteNumber * 8 + (i % 8)] = ((obstacleData[byteNumber] >> (i % 8)) & 1) != 0; // Due to the way bits are shifted into the bytes in the file, they must be shifted off in the opposite order hence the complicated expression for obstacles[...]. Right shift and AND with 1 takes that bit only

                if (i % 8 == 7)
                {
                    byteNumber++;
                }
            }
            return obstacles;
        }
    }
}
