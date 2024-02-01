using UserInterface.HelperClasses;
using System;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    internal class RelativePolToAbsoluteRect : IMultiValueConverter
    {
        /// <summary>
        /// Converts a relative polar coordinate to an asbolute rectangular one, relative to a specified canvas dimensions and optionally an origin.
        /// </summary>
        /// <param name="values">3 values: the <see cref="PolarPoint"/> to convert, the canvas width and the canvas height.</param>
        /// <param name="parameter">An optional origin, specified as a relative <see cref="Point"/> or <see cref="string"/>.</param>
        /// <returns>A relative polar coordinate.</returns>
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            RelativeDimension CoordinateConverter = new RelativeDimension();
            RectangularToPolar RecToPolConverter = new RectangularToPolar();

            if (values[0] is not PolarPoint || values[1] is not double || values[2] is not double)
            {
                return DependencyProperty.UnsetValue;
            }
            PolarPoint point = (PolarPoint)values[0];
            double canvasWidth = (double)values[1];
            double canvasHeight = (double)values[2];

            Point origin;
            if (parameter is null)
            {
                origin = new Point(0, 0);
            }
            else if (parameter is Point)
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

            Point relativePoint = (Point)RecToPolConverter.ConvertBack(point, targetType, origin, culture);
            return new Point(
                (double)CoordinateConverter.ConvertBack(canvasWidth, targetType, relativePoint.X.ToString(), culture),
                (double)CoordinateConverter.ConvertBack(canvasHeight, targetType, relativePoint.Y.ToString(), culture));
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Use AbsoluteRectToRelativePol instead.");
        }
    }
}
