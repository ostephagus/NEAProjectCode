using System;

namespace UserInterface.HelperClasses
{
    /// <summary>
    /// Abstract base class for spline calculators, allowing easy swapping of calculation methods.
    /// </summary>
    /// <typeparam name="PointT">The type of points that will be used for calculations.</typeparam>
    public abstract class SplineCalculator<PointT>
    {
        /// <summary>
        /// Adds or subtracts in modulo arithmetic, keeping the number between 0 and <c><paramref name="comparison"/> - 1</c>.
        /// </summary>
        /// <param name="number">The input.</param>
        /// <param name="numToAdd">The number to be added, or a negative number for subtraction.</param>
        /// <param name="comparison">The number that the output must be less than (not equal to).</param>
        /// <returns>The output of the modulo addition/subtraction.</returns>
        protected static int ModuloAdd(int number, int numToAdd, int comparison)
        {
            int moduloOutput = (number + numToAdd) % comparison;
            while (moduloOutput < 0)
            {
                moduloOutput += comparison;
            }
            return moduloOutput;
        }

        public abstract bool IsValidSpline { get; protected set; }

        public SplineCalculator() { }

        /// <summary>
        /// Adds a new control point.
        /// </summary>
        /// <param name="point">The point to add.</param>
        public abstract void AddControlPoint(PointT point);

        /// <summary>
        /// Replaces a control point.
        /// </summary>
        /// <param name="oldPoint">The existing control point to replace.</param>
        /// <param name="newPoint">The new control point that should be added instead.</param>
        public abstract void ModifyControlPoint(PointT oldPoint, PointT newPoint);

        /// <summary>
        /// Removes a control point. If the control point does not exist, does nothing.
        /// </summary>
        /// <param name="point">The point to remove.</param>
        /// <exception cref="InvalidOperationException">Thrown if there were fewer than 3 points when the method was called.</exception>
        public abstract void RemoveControlPoint(PointT point);

        /// <summary>
        /// Returns the point <paramref name="splineProgress"/> of the way through the spline, for rendering.
        /// </summary>
        /// <param name="splineProgress">A parameter from 0 to 1 representing how far through the spline the point is.</param>
        public abstract PointT CalculatePoint(double splineProgress);
    }
}
