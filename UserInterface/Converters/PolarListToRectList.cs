using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using UserInterface.HelperClasses;

namespace UserInterface.Converters
{
    public class PolarListToRectList : IMultiValueConverter
    {
        /// <summary>
        /// Converts a list of polar coordinates to a list of rectangular coordinates with a specified origin.
        /// </summary>
        /// <param name="values">An array of: polar point observable collection, parent width, parent height.</param>
        /// <param name="parameter">The origin, as fractions of the canvas size. Either a <see cref="Point"/> or <see cref="string"/> that can be converted to a point.</param>
        /// <returns>A <see cref="PointCollection"/> of rectangular points.</returns>
        /// <exception cref="NotImplementedException"></exception>
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            RectangularToPolar RectToPolConverter = new RectangularToPolar();
            VisualisationXCoordinate XCoordConverter = new VisualisationXCoordinate();
            VisualisationYCoordinate YCoordConverter = new VisualisationYCoordinate();

            if (values[0] is not ObservableCollection<PolarPoint> || values[1] is not double || values[2] is not double)
            {
                return DependencyProperty.UnsetValue;
            }
            ObservableCollection<PolarPoint> polarPoints = (ObservableCollection<PolarPoint>)values[0];
            double parentWidth = (double)values[1];
            double parentHeight = (double)values[2];

            PointCollection points = new PointCollection();
            foreach (PolarPoint point in polarPoints)
            {
                Point relativePoint = (Point)RectToPolConverter.ConvertBack(point, targetType, parameter, culture); // This has its dimensions between 0 and 1.
                Point absolutePoint = new Point((double)XCoordConverter.Convert(parentWidth, targetType, relativePoint.X.ToString(), culture), (double)YCoordConverter.Convert(parentHeight, targetType, relativePoint.Y.ToString(), culture));
                points.Add(absolutePoint);
            }
            return points;
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Conversion not allowed.");
        }
    }
}
