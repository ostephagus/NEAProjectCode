using System;

namespace UserInterface.HelperClasses
{
    /// <summary>
    /// Represents a point defined by polar coordinates.
    /// </summary>
    public class PolarPoint : IComparable<PolarPoint>, IEquatable<PolarPoint>
    {
        /// <summary>
        /// The distance from the origin to the point.
        /// </summary>
        public double Radius;

        /// <summary>
        /// The angle, in radians, with respect to a right-facing initial line.
        /// </summary>
        public double Angle;

        /// <summary>
        /// The angle, in degrees, with respect to a right-facing initial line.
        /// </summary>
        public double DegreesAngle { get => Angle * 180 / Math.PI; }

        /// <summary>
        /// Creates a polar point with a given radius and angle.
        /// </summary>
        /// <param name="radius">The distance from the origin to the point.</param>
        /// <param name="angle">The angle, in radians, with respect to a right-facing initial line.</param>
        public PolarPoint(double radius, double angle)
        {
            Radius = radius;
            Angle = angle;
        }

        public int CompareTo(PolarPoint? other)
        {
            if (Angle == other?.Angle) // If same angle, sort on radius
            {
                return Radius.CompareTo(other.Radius);
            }
            return Angle.CompareTo(other?.Angle);
        }

        public bool Equals(PolarPoint? other)
        {
            return Radius == other?.Radius && Angle == other?.Angle;
        }

        public override string ToString()
        {
            return $"{Radius}, {Angle}";
        }
    }
}
