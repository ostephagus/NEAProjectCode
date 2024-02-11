using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace UserInterface.Converters
{
    public class Units : IValueConverter
    {
        private readonly UnitClasses.Unit? unit;

        private UnitClasses.Unit Unit { get => unit ?? new UnitClasses.Dimensionless(); }

        public Units(UnitClasses.Unit? unit)
        {
            this.unit = unit;
        }

        /// <summary>
        /// Converts a non-SI unit to the corresponding SI unit.
        /// </summary>
        /// <param name="value">The value in the converter's non-SI unit.</param>
        /// <returns><paramref name="value"/>, converted to the corresponding SI unit.</returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not double nonSIValue)
            {
                return DependencyProperty.UnsetValue;
            }

            return nonSIValue * Unit.ConversionRatio;
        }

        /// <summary>
        /// Converts an SI unit to a specified non-SI unit.
        /// </summary>
        /// <param name="value">The value in the SI unit.</param>
        /// <returns><paramref name="value"/>, converted to the converter's non-SI unit.</returns>
        /// <exception cref="NotImplementedException"></exception>
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is not double SIValue)
            {
                return DependencyProperty.UnsetValue;
            }

            return SIValue / Unit.ConversionRatio;
        }
    }
}
