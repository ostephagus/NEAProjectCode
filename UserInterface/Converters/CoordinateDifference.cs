using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    public abstract class CoordinateDifference : IValueConverter
    {
        protected readonly VisualisationCoordinate VisualisationCoordinateConverter;

        protected abstract double FindLength(double start, double end);

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not double || parameter is not string || !((string)parameter).Contains(','))
            {
                return DependencyProperty.UnsetValue;
            }

            string parameterNoSpaces = new(((string)parameter).ToCharArray().Where(c => !char.IsWhiteSpace(c)).ToArray());
            string[] parameters = parameterNoSpaces.Split(',');

            if (!double.TryParse(parameters[0], out double elementStartCoord) || !double.TryParse(parameters[1], out double elementEndCoord))
            {
                return DependencyProperty.UnsetValue;
            }

            double elementLength = FindLength(elementStartCoord, elementEndCoord); // Use formula to get relative length of element
            return VisualisationCoordinateConverter.Convert(value, targetType, elementLength.ToString(), culture); // Convert this to actual coordinates
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException();
        }

        public CoordinateDifference(VisualisationCoordinate VisualisationCoordinateConverter) // Derived classes will need to instantiate VisualisationCoordinateConverter in their constructors.
        {
            this.VisualisationCoordinateConverter = VisualisationCoordinateConverter;
        }
    }
}
