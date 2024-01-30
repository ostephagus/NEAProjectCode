using System;
using System.Globalization;
using System.Windows;

namespace UserInterface.Converters
{
    public abstract class VisualisationCoordinate
    {
        private readonly RelativeDimension RelativeDimensionConverter;

        public abstract double TranslateVisualisationCoordinate(double p);

        /// <summary>
        /// Converts a canvas dimension and fraction to a coordinate as displayed by the visualisation.
        /// </summary>
        /// <param name="value">The canvas dimension.</param>
        /// <param name="parameter">The fraction of the canvas to use.</param>
        /// <returns></returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (parameter is not string || !double.TryParse((string)parameter, out double fractionOfCanvas))
            {
                return DependencyProperty.UnsetValue;
            }
            fractionOfCanvas = TranslateVisualisationCoordinate(fractionOfCanvas);

            return RelativeDimensionConverter.Convert(value, targetType, fractionOfCanvas.ToString(), culture);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException();
        }

        public VisualisationCoordinate()
        {
            RelativeDimensionConverter = new RelativeDimension();
        }
    }
}