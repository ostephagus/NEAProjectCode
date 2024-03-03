using System.IO;

namespace UserInterface.HelperClasses
{
    public class ObstacleHolder
    {
        private string? fileName;
        private bool usingObstacleFile;

        public string? FileName { get => fileName; set => fileName = value; }
        public bool UsingObstacleFile { get => usingObstacleFile; set => usingObstacleFile = value; }

        public ObstacleHolder(string? fileName, bool usingObstacleFile)
        {
            this.fileName = fileName;
            this.usingObstacleFile = usingObstacleFile;
        }
    }
}
