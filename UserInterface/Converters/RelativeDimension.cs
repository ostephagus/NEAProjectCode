using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    /// <summary>
    /// Converts between relative (0-1) and absolute coordinates.
    /// </summary>
    public class RelativeDimension : IValueConverter
    {
        /// <summary>
        /// Converts a number between 0 and 1 into a fraction of the dimension of the parent.
        /// </summary>
        /// <param name="value">The dimension of the parent.</param>
        /// <param name="parameter">The relative coordinate.</param>
        /// <returns>The value that represents the fraction of the parent dimension.</returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not double || parameter is not string || !double.TryParse((string)parameter, out double fractionOfParent))
            {
                return DependencyProperty.UnsetValue;
            }
            double parentDimension = (double)value;
            return fractionOfParent * parentDimension;
        }

        /// <summary>
        /// Converts an absolute dimension to a number between 0 and 1 relative to the dimension of the parent.
        /// </summary>
        /// <param name="value">The dimension of the parent.</param>
        /// <param name="parameter">The absolute coordinate.</param>
        /// <returns>A relative coordinate between 0 and 1</returns>
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not double || parameter is not string || !double.TryParse((string)parameter, out double absoluteCoordinate))
            {
                return DependencyProperty.UnsetValue;
            }
            double parentDimension = (double)value;
            return absoluteCoordinate / parentDimension;
        }
    }
}
