namespace FileMakerHardCoded
{
    public static class FileMaker
    {
        private static bool[] CreateObstacleArray(int xLength, int yLength, Predicate<(int, int)> ObstacleDefiningFunction)
        {
            bool[] obstacleArray = new bool[xLength * yLength];
            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    obstacleArray[i * yLength + j] = ObstacleDefiningFunction((i, j));
                }
            }
            return obstacleArray;
        }

        private static void WriteFile(string filePath, bool[] obstacleArray, int xLength, int yLength)
        {
            byte[] buffer = new byte[obstacleArray.Length / 8 + (obstacleArray.Length % 8 == 0 ? 0 : 1)]; // Divide the length by 8 and add one if the length does not divide evenly. Also add 1 byte for FLDEND
            
            int index = 0;
            for (int i = 0; i < obstacleArray.Length; i++)
            {
                buffer[index] |= (byte)((obstacleArray[i] ? 1 : 0) << i % 8); // Convert the bool to 1 or 0, shift it left the relevant amount of times and OR it with the current value in the buffer
                if (i % 8 == 7) // Add one to the index if the byte is full
                {
                    index++;
                }
            }

            using Stream stream = File.OpenWrite(filePath);
            using BinaryWriter writer = new BinaryWriter(stream);

            writer.Write(xLength - 2); // Subtract 2 for iMax and jMax.
            writer.Write(yLength - 2);
            writer.Write(buffer);
        }

        public static void CreateFile(string filePath, int xLength, int yLength, Predicate<(int, int)> ObstacleDefiningFunction)
        {
            bool[] obstacleArray = CreateObstacleArray(xLength, yLength, ObstacleDefiningFunction);
            WriteFile(filePath, obstacleArray, xLength, yLength);
        }
    }
}
