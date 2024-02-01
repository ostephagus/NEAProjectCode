using System.Windows.Data;

namespace UserInterface.Converters
{
    public class VisualisationXCoordinate : VisualisationCoordinate, IValueConverter
    {
        /// <summary>
        /// Translates an x coordinate from [0, 1] to the precise point as rendered by <see cref="VisualisationControl" />.
        /// </summary>
        /// <param name="p">The x coordinate in [0, 1] to translate.</param>
        /// <returns>A point that maps to the <see cref="VisualisationControl"/> space.</returns>
        public override double TranslateVisualisationCoordinate(double p)
        {
            return -0.000451565 + 1.01036 * p;
        }

        /// <summary>
        /// Translates an x coordinate from the precise point as rendered by <see cref="VisualisationControl" /> to [0, 1].
        /// </summary>
        /// <param name="p">The point that maps to the <see cref="VisualisationControl"/> space to translate.</param>
        /// <returns>An x coordinate in [0, 1].</returns>
        public override double TranslateCanvasCoordinate(double p)
        {
            return (p + 0.000451565) / 1.01036;
        }
    }
}
