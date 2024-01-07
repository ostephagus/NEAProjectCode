using System;
using System.Windows.Controls.Primitives;
using System.Windows.Data;

namespace UserInterface.HelperClasses
{
    class BoolToTickPlacementConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value is not bool)
            {
                return TickPlacement.None;
            }
            if ((bool)value)
            {
                return TickPlacement.BottomRight;
            }
            return TickPlacement.None;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new InvalidOperationException("Conversion not allowed");
        }
    }
}
