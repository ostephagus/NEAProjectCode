using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    public class AbsoluteToRelativeRect : IMultiValueConverter
    {
        /// <summary>
        /// Converts an absolute rectangular coordinate to a relative rectangular coordinate, optionally flipping the y coordinate to make the origin bottom-left rather than bottom-right.
        /// </summary>
        /// <param name="values">An array of objects in the form: point, parent width, parent height.</param>
        /// <param name="parameter">A boolean specifying whether to flip the y coordinate.</param>
        /// <returns>A relative rectangular coordinate.</returns>
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            RelativeDimension CoordinateConverter = new RelativeDimension();

            if (values[0] is not Point || values[1] is not double || values[2] is not double)
            {
                return DependencyProperty.UnsetValue;
            }
            Point point = (Point)values[0];
            double canvasWidth = (double)values[1];
            double canvasHeight = (double)values[2];

            bool flipY;
            if (parameter is null)
            {
                flipY = false;
            }
            else if (parameter is bool)
            {
                flipY = (bool)parameter;
            }
            else if (parameter is string)
            {
                if (!bool.TryParse(((string)parameter).ToLower(), out flipY))
                {
                    return DependencyProperty.UnsetValue;
                }
            }
            else
            {
                return DependencyProperty.UnsetValue;
            }

            double xCoordinate = point.X * 2;
            double yCoordinate = point.Y * 2;

            double relativeX = (double)CoordinateConverter.ConvertBack(canvasWidth, targetType, xCoordinate.ToString(), culture);
            double relativeY = (double)CoordinateConverter.ConvertBack(canvasHeight, targetType, yCoordinate.ToString(), culture);
            if (flipY)
            {
                relativeY = 1 - relativeY;
            }

            return new Point(relativeX, relativeY);
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Cannot convert back.");
        }
    }
}
