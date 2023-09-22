using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls.Primitives;
using System.Windows.Data;

namespace UserInterface
{
    public class ValueConverters
    {
        public class BoolToTickStatus : IValueConverter
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
}
