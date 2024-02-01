using System.Windows.Data;

namespace UserInterface.Converters
{
    public class VisualisationYCoordinate : VisualisationCoordinate, IValueConverter
    {
        /// <summary>
        /// Translates a y coordinate from [0, 1] to the precise point as rendered by <see cref="VisualisationControl" />.
        /// </summary>
        /// <param name="p">The y coordinate in [0, 1] to translate.</param>
        /// <returns>A point that maps to the <see cref="VisualisationControl"/> space.</returns>
        public override double TranslateVisualisationCoordinate(double p)
        {
            return 1.009 - 1.0099 * p;
        }

        /// <summary>
        /// Translates a y coordinate from the precise point as rendered by <see cref="VisualisationControl" /> to [0, 1].
        /// </summary>
        /// <param name="p">The point that maps to the <see cref="VisualisationControl"/> space to translate.</param>
        /// <returns>A y coordinate in [0, 1].</returns>
        public override double TranslateCanvasCoordinate(double p)
        {
            return (1.009 - p) / 1.0099;
        }
    }
}
