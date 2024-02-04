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
        /// <param name="values">An array of polar point observable collection, and origin (as fractions of the canvas size. Either a <see cref="Point"/> or <see cref="string"/> that can be converted to a point).</param>
        /// <returns>A <see cref="PointCollection"/> of rectangular points.</returns>
        /// <exception cref="NotImplementedException"></exception>
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            RectangularToPolar RectToPolConverter = new RectangularToPolar();

            if (values[0] is not ObservableCollection<PolarPoint>)
            {
                return DependencyProperty.UnsetValue;
            }
            ObservableCollection<PolarPoint> polarPoints = (ObservableCollection<PolarPoint>)values[0];
            object origin = values[1]; // Allow RectToPolConverter to do the conversion

            PointCollection points = new PointCollection();
            foreach (PolarPoint point in polarPoints)
            {
                Point rectangularPoint = (Point)RectToPolConverter.ConvertBack(point, targetType, origin, culture);
                points.Add(new Point(rectangularPoint.X, 100 - rectangularPoint.Y)); // Flip the y coordinates.
            }
            return points;
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Conversion not allowed.");
        }
    }
}
