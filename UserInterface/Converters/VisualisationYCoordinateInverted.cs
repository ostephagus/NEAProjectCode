using System.Windows.Data;

namespace UserInterface.Converters
{
    public class VisualisationYCoordinateInverted : VisualisationCoordinate, IValueConverter
    {
        /// <summary>
        /// Translates a y coordinate from [0, 1] to the precise point as rendered by <see cref="VisualisationControl" />.
        /// </summary>
        /// <param name="p">The y coordinate in [0, 1] to translate.</param>
        /// <returns>A point that maps to the <see cref="VisualisationControl"/> space.</returns>
        public override double TranslateVisualisationCoordinate(double p)
        {
            return -0.0009 + 1.0099 * p;
        }

        public override double TranslateCanvasCoordinate(double p)
        {
            return (p + 0.0009) / 1.0099;
        }
    }
}
