using System;
using System.Globalization;
using System.Windows;

namespace UserInterface.Converters
{
    public abstract class VisualisationCoordinate
    {
        private readonly RelativeDimension RelativeDimensionConverter;

        public abstract double TranslateVisualisationCoordinate(double p);

        public abstract double TranslateCanvasCoordinate(double p);

        /// <summary>
        /// Converts a canvas dimension and fraction to a coordinate as displayed by the visualisation.
        /// </summary>
        /// <param name="value">The canvas dimension.</param>
        /// <param name="parameter">The fraction of the canvas to use.</param>
        /// <returns>An absolute coordinate that aligns with coordinates of the visualisation.</returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (parameter is not string || !double.TryParse((string)parameter, out double fractionOfCanvas))
            {
                return DependencyProperty.UnsetValue;
            }
            fractionOfCanvas = TranslateVisualisationCoordinate(fractionOfCanvas);

            return RelativeDimensionConverter.Convert(value, targetType, fractionOfCanvas.ToString(), culture);
        }


        /// <summary>
        /// Converts an absolute coordinate as displayed by the visualisation to a fraction of the canvas.
        /// </summary>
        /// <param name="value">The canvas dimension.</param>
        /// <param name="parameter">The absolute coordinate that aligns with the visualisation.</param>
        /// <returns>A coordinate relative to the canvas [0, 1].</returns>
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            // Input validation is done by RelativeDimensionConverter
            double relativeCoordinate = (double)RelativeDimensionConverter.ConvertBack(value, targetType, parameter, culture);

            return TranslateCanvasCoordinate(relativeCoordinate);
        }

        public VisualisationCoordinate()
        {
            RelativeDimensionConverter = new RelativeDimension();
        }
    }
}