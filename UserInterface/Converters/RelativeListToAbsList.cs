using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace UserInterface.Converters
{
    public class RelativeListToAbsList : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values[0] is not ObservableCollection<Point> relativeList)
            {
                return DependencyProperty.UnsetValue;
            }

            Point origin;
            if (values[1] is Point)
            {
                origin = (Point)values[1];
            }
            else if (values[1] is string)
            {
                string parameterNoSpaces = new(((string)parameter).ToCharArray().Where(c => !char.IsWhiteSpace(c)).ToArray());
                string[] parameters = parameterNoSpaces.Split(',');

                if (!double.TryParse(parameters[0], out double x) || !double.TryParse(parameters[1], out double y))
                {
                    return DependencyProperty.UnsetValue;
                }
                origin = new Point(x, y);
            }

            PointCollection absPoints = new PointCollection();

            foreach (Point point in relativeList)
            {
                Point absUnflippedPoint = point + (Vector)origin;
                absPoints.Add(new Point(absUnflippedPoint.X, 100 - absUnflippedPoint.Y));
            }
            return absPoints;
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new InvalidOperationException("Conversion not allowed.");
        }
    }
}
