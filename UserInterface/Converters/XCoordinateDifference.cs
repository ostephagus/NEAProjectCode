namespace UserInterface.Converters
{
    public class XCoordinateDifference : CoordinateDifference
    {
        /// <summary>
        /// Translates an x coordinate from [0, 1] to the precise point as rendered by <see cref="VisualisationControl" />.
        /// </summary>
        /// <param name="p">The x coordinate in [0, 1] to translate.</param>
        /// <returns>A point that maps to the <see cref="VisualisationControl"/> space.</returns>
        private double TranslateVisualisationCoordinate(double p)
        {
            return -0.000451565 + 1.01036 * p;
        }

        protected override double FindLength(double start, double end)
        {
            return TranslateVisualisationCoordinate(end) - TranslateVisualisationCoordinate(start) + 0.0006;
        }

        public XCoordinateDifference() : base(new VisualisationXCoordinate()) { }
    }
}
