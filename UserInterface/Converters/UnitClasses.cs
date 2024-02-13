using System;

namespace UserInterface.Converters
{
    /// <summary>
    /// Static constant-containing class that defines the conversion ratios to convert from the non-SI unit to the corresponding SI unit. For example, to convert a value in feet to a value in metres, multiply the value in feet by <see cref="LengthMultipliers.Feet"/>.
    /// </summary>
    public class UnitClasses // All values to 6 sig fig
    {
        public enum UnitSystem
        {
            SI,
            CGS,
            DefaultImperial,
            OtherMetric,
            OtherImperial
        }

        /// <summary>
        /// Multipliers to convert length units to metres.
        /// </summary>
        private static class LengthMultipliers
        {
            public static readonly double Feet = 0.304800;
            public static readonly double Inches = 0.025400;
            public static readonly double Centimetres = 0.010000;
            public static readonly double Millimetres = 0.001000;
        }
        /// <summary>
        /// Multipliers to convert speed units to metres per second.
        /// </summary>
        private static class SpeedMultipliers
        {
            public static readonly double MilesPerHour = 0.447040;
            public static readonly double KilometresPerHour = 0.277778;
            public static readonly double CentimetresPerSecond = LengthMultipliers.Centimetres;
            public static readonly double FeetPerSecond = LengthMultipliers.Feet;
        }
        /// <summary>
        /// Multipliers to convert time units to seconds.
        /// </summary>
        private static class TimeMultipliers
        {
            public static readonly double Milliseconds = 0.001;
            public static readonly double Minutes = 60;
            public static readonly double Hours = 3600;
        }
        /// <summary>
        /// Multipliers to convert density units to kilograms per cubic metre.
        /// </summary>
        private static class DensityMultipliers
        {
            public static readonly double GramsPerCubicCentimetre = 1000.00;
            public static readonly double PoundsPerCubicInch = 27679.90;
            public static readonly double PoundsPerCubicFoot = 16.01846;
        }
        /// <summary>
        /// Multipliers to convert viscosity units to kilograms per metre per second.
        /// </summary>
        private static class ViscosityMultipliers
        {
            public static readonly double GramsPerCentimetrePerSecond = 0.1000000;
            public static readonly double PoundSecondsPerSquareFoot = 47.800000;
        }

        public abstract class Unit
        {
            /// <summary>
            /// Multiplier when converting from this unit to the corresponding SI unit.
            /// </summary>
            protected readonly double conversionRatio;
            protected readonly string longName;
            protected readonly string shortName;

            public double ConversionRatio => conversionRatio;
            public string LongName => longName;
            public string ShortName => shortName;
            public Unit(double conversionRatio, string longName, string shortName)
            {
                this.conversionRatio = conversionRatio;
                this.longName = longName;
                this.shortName = shortName;
            }
        }

        public class Dimensionless() : Unit(1, "", "");

        public abstract class LengthUnit(double conversionRatio, string longName, string shortName) : Unit(conversionRatio, longName, shortName) { }
        public abstract class SpeedUnit(double conversionRatio, string longName, string shortName) : Unit(conversionRatio, longName, shortName) { }
        public abstract class TimeUnit(double conversionRatio, string longName, string shortName) : Unit(conversionRatio, longName, shortName) { }
        public abstract class DensityUnit(double conversionRatio, string longName, string shortName) : Unit(conversionRatio, longName, shortName) { }
        public abstract class ViscosityUnit(double conversionRatio, string longName, string shortName) : Unit(conversionRatio, longName, shortName) { }

        public class Metre() : LengthUnit(1, "Metres", "m");
        public class Foot() : LengthUnit(LengthMultipliers.Feet, "Feet", "Ft");
        public class Inch() : LengthUnit(LengthMultipliers.Inches, "Inches", "In");
        public class Centimetre() : LengthUnit(LengthMultipliers.Centimetres, "Centimetres", "cm");
        public class Millimetre() : LengthUnit(LengthMultipliers.Millimetres, "Millimetres", "mm");

        public class MetrePerSecond() : SpeedUnit(1, "Metres per second", "m/s");
        public class CentimetrePerSecond() : SpeedUnit(SpeedMultipliers.CentimetresPerSecond, "Centimetres per second", "cm/s");
        public class MilePerHour() : SpeedUnit(SpeedMultipliers.MilesPerHour, "Miles per hour", "mph");
        public class KilometrePerHour() : SpeedUnit(SpeedMultipliers.KilometresPerHour, "Kilometres per hour", "km/h");
        public class FootPerSecond() : SpeedUnit(SpeedMultipliers.FeetPerSecond, "Feet per second", "Ft/s");

        public class Second() : TimeUnit(1, "Seconds", "s");
        public class Millisecond() : TimeUnit(TimeMultipliers.Milliseconds, "Milliseconds", "ms");
        public class Minute() : TimeUnit(TimeMultipliers.Minutes, "Minutes", "min");
        public class Hour() : TimeUnit(TimeMultipliers.Hours, "Hours", "hr");

        public class KilogramPerCubicMetre() : DensityUnit(1, "Kilograms per cubic metre", "kg/m³");
        public class GramPerCubicCentimetre() : DensityUnit(DensityMultipliers.GramsPerCubicCentimetre, "Grams per cubic centimetre", "g/cm³");
        public class PoundPerCubicInch() : DensityUnit(DensityMultipliers.PoundsPerCubicInch, "Pounds per cubic inch", "lb/in³");
        public class PoundPerCubicFoot() : DensityUnit(DensityMultipliers.PoundsPerCubicFoot, "Pounds per cubic foot", "lb/ft³");

        public class KilogramPerMetrePerSecond() : ViscosityUnit(1, "Kilograms per metre per second", "kg/m s");
        public class GramPerCentimetrePerSecond() : ViscosityUnit(ViscosityMultipliers.GramsPerCentimetrePerSecond, "Grams per centimetre per second", "g/cm s");
        public class PoundSecondPerSquareFoot() : ViscosityUnit(ViscosityMultipliers.PoundSecondsPerSquareFoot, "Pound-seconds per square foot", "lb·s/ft²");

        public class UnitSystemList
        {
            public readonly UnitSystem unitSystem;
            public readonly LengthUnit lengthUnit;
            public readonly SpeedUnit speedUnit;
            public readonly TimeUnit timeUnit;
            public readonly DensityUnit densityUnit;
            public readonly ViscosityUnit viscosityUnit;

            public UnitSystemList(UnitSystem unitSystem, LengthUnit lengthUnit, SpeedUnit speedUnit, TimeUnit timeUnit, DensityUnit densityUnit, ViscosityUnit viscosityUnit)
            {
                this.unitSystem = unitSystem;
                this.lengthUnit = lengthUnit;
                this.speedUnit = speedUnit;
                this.timeUnit = timeUnit;
                this.densityUnit = densityUnit;
                this.viscosityUnit = viscosityUnit;
            }
        }
    }
}
