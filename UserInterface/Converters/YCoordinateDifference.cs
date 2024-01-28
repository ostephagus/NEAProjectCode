using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UserInterface.Converters
{
    internal class YCoordinateDifference : CoordinateDifference
    {
        /// <summary>
        /// Translates a y coordinate from [0, 1] to the precise point as rendered by <see cref="VisualisationControl" />.
        /// </summary>
        /// <param name="p">The y coordinate in [0, 1] to translate.</param>
        /// <returns>A point that maps to the <see cref="VisualisationControl"/> space.</returns>
        public double TranslateVisualisationCoordinate(double p)
        {
            return 1.009 - 1.0099 * p;
        }

        protected override double FindLength(double start, double end)
        {
            return TranslateVisualisationCoordinate(end) - TranslateVisualisationCoordinate(start) + 0.0017;
        }

        public YCoordinateDifference() : base(new VisualisationYCoordinateInverted()) { }
    }
}
