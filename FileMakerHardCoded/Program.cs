namespace FileMakerHardCoded
{
    public class Program
    {
        private const int iMax = 200;
        private const int jMax = 100;
        private const string filePath = "square.simobst";

        private static bool ObstacleDefiningFunction((int, int) pointTuple)
        {
            // 1 is an obstacle cell, 0 is a fluid cell.
            (int x, int y) = pointTuple;
            return x > iMax * 0.45 && x < iMax * 0.55 && y > jMax * 0.45 && y < jMax * 0.55;
        }

        static void Main(string[] args)
        {
            FileMaker.CreateFile(filePath, iMax + 2, jMax + 2, ObstacleDefiningFunction);
        }
    }
}
