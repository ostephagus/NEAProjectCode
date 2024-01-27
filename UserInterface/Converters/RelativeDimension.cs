using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    /// <summary>
    /// Converts a number between 0 and 1 into a fraction of the dimension of the parent.
    /// </summary>
    public class RelativeDimension : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not double || parameter is not string || !double.TryParse((string)parameter, out double fractionOfParent))
            {
                return DependencyProperty.UnsetValue;
            }
            double parentDimension = (double)value;
            return fractionOfParent * parentDimension;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException();
        }
    }
}
