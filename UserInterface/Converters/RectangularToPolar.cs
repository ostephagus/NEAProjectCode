using System;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;
using UserInterface.HelperClasses;

namespace UserInterface.Converters
{
    /// <summary>
    /// Converter class that can convert rectangular to polar and back again with respect to a given pole.
    /// </summary>
    public class RectangularToPolar : IValueConverter
    {
        /// <summary>
        /// Converts a rectangular coordinate to a polar coordinate, with the pole to use optionally specified in rectangular coordinates in <paramref name="parameter"/>
        /// </summary>
        /// <param name="value">The rectangular coordinate, as a <see cref="Point"/>.</param>
        /// <param name="parameter">The pole to use in the conversion (as a rectangular <see cref="Point"/> or <see cref="string"/>), or <see cref="null"/> to use (0, 0).</param>
        /// <returns>A <see cref="PolarPoint"/> representing the converted coordinate.</returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not Point || !(parameter is Point || parameter is string || parameter is null))
            {
                return DependencyProperty.UnsetValue;
            }

            Point point = (Point)value;
            Point origin;
            if (parameter is Point)
            {
                origin = (Point)parameter;
            }
            else if (parameter is string)
            {
                string parameterNoSpaces = new(((string)parameter).ToCharArray().Where(c => !char.IsWhiteSpace(c)).ToArray());
                string[] parameters = parameterNoSpaces.Split(',');

                if (!double.TryParse(parameters[0], out double x) || !double.TryParse(parameters[1], out double y))
                {
                    return DependencyProperty.UnsetValue;
                }
                origin = new Point(x, y);
            }
            else // parameter is null
            {
                origin = new Point(0, 0); // Origin not specified - use default.
            }

            Vector distFromOrigin = point - origin;

            return new PolarPoint(distFromOrigin.Length, Math.Atan2(distFromOrigin.Y, distFromOrigin.X)); // This is the only line that actually converts a rectangular coordinate to a polar one.
        }

        /// <summary>
        /// Converts a polar coordinate to a rectangular coordinate, with the pole to use optionally specified in rectangular coordinates in <paramref name="parameter"/>
        /// </summary>
        /// <param name="value">The polar coordinate, as a <see cref="PolarPoint"/>.</param>
        /// <param name="parameter">The pole to use in the conversion (as a rectangular <see cref="Point"/> or <see cref="string"/>), or <see cref="null"/> to use (0, 0).</param>
        /// <returns>A <see cref="Point"/> representing the converted coordinate.</returns>
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not PolarPoint || !(parameter is Point || parameter is string || parameter is null))
            {
                return DependencyProperty.UnsetValue;
            }

            PolarPoint point = (PolarPoint)value;
            Point origin;
            if (parameter is Point)
            {
                origin = (Point)parameter;
            }
            else if (parameter is string)
            {
                string parameterNoSpaces = new(((string)parameter).ToCharArray().Where(c => !char.IsWhiteSpace(c)).ToArray());
                string[] parameters = parameterNoSpaces.Split(',');

                if (!double.TryParse(parameters[0], out double x) || !double.TryParse(parameters[1], out double y))
                {
                    return DependencyProperty.UnsetValue;
                }
                origin = new Point(x, y);
            }
            else // parameter is null
            {
                origin = new Point(0, 0); // Origin not specified - use default.
            }

            Vector distanceFromOrigin = new Vector(point.Radius * Math.Cos(point.Angle), point.Radius * Math.Sin(point.Angle)); // Convert the polar coordinate to a rectangular vector.

            return origin + distanceFromOrigin; // Translate the origin by the vector to get the final rectangular point.
        }
    }
}
