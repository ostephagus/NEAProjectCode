using System;
using System.Globalization;
using System.Windows;

namespace UserInterface.Converters
{
    public abstract class VisualisationCoordinate
    {
        private readonly RelativeDimension RelativeDimensionConverter;

        public abstract double TranslateVisualisationCoordinate(double p);

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