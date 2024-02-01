using System;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    public class AbsoluteRectToRelativePol : IMultiValueConverter
    {
        /// <summary>
        /// Converts an absolute rectangular coordinate to a relative polar one, relative to a specified canvas dimensions and optionally an origin.
        /// </summary>
        /// <param name="values">3 values: the <see cref="Point"/> to convert, the canvas width and the canvas height.</param>
        /// <param name="parameter">An optional origin, specified as a relative <see cref="Point"/> or <see cref="string"/>.</param>
        /// <returns>A relative polar coordinate.</returns>
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            RelativeDimension CoordinateConverter = new RelativeDimension();
            RectangularToPolar RecToPolConverter = new RectangularToPolar();

            if (values[0] is not Point || values[1] is not double || values[2] is not double)
            {
                return DependencyProperty.UnsetValue;
            }
            Point point = (Point)values[0];
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

            double xCoordinate = point.X * 2;
            double yCoordinate = point.Y * 2;


            Point relativePoint = new Point(
                (double)CoordinateConverter.ConvertBack(canvasWidth, targetType, xCoordinate.ToString(), culture),
                1 - (double)CoordinateConverter.ConvertBack(canvasHeight, targetType, yCoordinate.ToString(), culture)); // Flip the y coordinate about the centre of the canvas to make it 0 at the bottom rather than at the top.
            return RecToPolConverter.Convert(relativePoint, targetType, origin, culture);
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Use RelativePolToAbsoluteRect instead.");
        }
    }
}
