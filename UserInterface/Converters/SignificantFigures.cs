using System;
using System.Globalization;
using System.Windows.Data;

namespace UserInterface.Converters
{
    [ValueConversion(typeof(float), typeof(string))]
    public class SignificantFigures : IValueConverter
    {
        /// <summary>
        /// Rounds an input value to a number of significant figures, returning a string to be displayed.
        /// </summary>
        /// <param name="value">The value, as a <see cref="float"/>, to round.</param>
        /// <param name="parameter">The number of significant figures to round to, as a <see cref="string"/> representation of an int.</param>
        /// <returns>The rounded value, cast from a <see cref="string"/> to an <see cref="object"/>.</returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            // Input validation
            if (!int.TryParse((string)parameter, out int iParameter))
            {
                return "";
            }
            float fValue;
            if (value is double dValue)
            {
                fValue = (float)dValue;
            }
            else if (value is float)
            {
                fValue = (float)value;
            }
            else
            {
                return "";
            }


            return fValue.ToString($"G{iParameter}"); // Use the ToString method with the number of SF as the parameter to it.
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Values cannot be converted back once they have been rounded.");
        }
    }
}
