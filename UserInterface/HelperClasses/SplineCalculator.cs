namespace UserInterface.HelperClasses
{
    /// <summary>
    /// Abstract base class for spline calculators, allowing easy swapping of calculation methods.
    /// </summary>
    /// <typeparam name="PointT">The type of points that will be used for calculations.</typeparam>
    public abstract class SplineCalculator<PointT>
    {
        public abstract bool IsValidSpline { get; protected set; }

        public SplineCalculator() { }

        public abstract void AddControlPoint(PointT point);

        public abstract void ModifyControlPoint(PointT oldPoint, PointT newPoint);

        public abstract void RemoveControlPoint(PointT point);

        /// <summary>
        /// Returns the point <paramref name="splineProgress"/> of the way through the spline, for rendering.
        /// </summary>
        /// <param name="splineProgress">A parameter from 0 to 1 representing how far through the spline the point is.</param>
        public abstract PointT CalculatePoint(double splineProgress);
    }
}
