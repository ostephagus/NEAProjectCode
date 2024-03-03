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
            dataWidth = 100;
            dataHeight = 100;
        }
    }
}
