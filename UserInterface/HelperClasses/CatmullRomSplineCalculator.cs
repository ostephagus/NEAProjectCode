using System.Windows;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace UserInterface.HelperClasses
{
    public class CatmullRomSplineCalculator : SplineCalculator<Point>
    {
        private readonly double[] definingMatrixData = [0, -0.5, 1, -0.5, 1, 0, -2.5, 1.5, 0, 0.5, 2, -1.5, 0, 0, -0.5, 0.5]; // Defining matrix for Catmull-Rom stored in column-major order

        private List<Point> controlPoints;
        private Matrix<double> definingMatrix;
        private bool isValidSpline;

        /// <summary>
        /// Non-static partial application of <see cref="SplineCalculator{PointT}.ModuloAdd(int, int, int)"/> wrapping based on the length of <see cref="controlPoints"/>.
        /// </summary>
        private int WrappingIndexAdd(int index, int numToAdd) => ModuloAdd(index, numToAdd, controlPoints.Count);

        public override bool IsValidSpline { get => isValidSpline; protected set => isValidSpline = value; }

        /// <summary>
        /// Interpolates a single coordinate using the Catmull-Rom method between two points <paramref name="p1"/> and <paramref name="p2"/>. The coordinates of the points <paramref name="p0"/> and <paramref name="p3"/> either side are also needed.
        /// </summary>
        /// <param name="t">The "time" parameter, from 0 to 1, representing the progress along the spline</param>
        /// <returns>The coordinate, interpolated along the spline.</returns>
        private double Interpolate(double t, double p0, double p1, double p2, double p3)
        {
            Matrix<double> times = Matrix<double>.Build.Dense(4, 1, [1.0, t, t * t, t * t * t]);

            Vector<double> points = Vector<double>.Build.Dense([p0, p1, p2, p3]);

            return (times * definingMatrix * points)[0]; // This returns a 1-D vector, so just return the single value within.
        }

        public CatmullRomSplineCalculator() : base() 
        {
            controlPoints = new List<Point>();
            definingMatrix = Matrix<double>.Build.Dense(4, 4, definingMatrixData);
        }

        public override void AddControlPoint(Point point)
        {
            throw new NotImplementedException("Need to figure out how to add control points in the correct place in list of control points");
        }

        public override void ModifyControlPoint(Point oldPoint, Point newPoint)
        {
            controlPoints.Remove(oldPoint);
            controlPoints.Add(newPoint);
            throw new NotImplementedException("Need to figure out how to add control points in the correct place in list of control points");
        }

        public override void RemoveControlPoint(Point point)
        {
            if (controlPoints.Count < 3)
            {
                throw new InvalidOperationException("A spline must have at least 3 points.");
            }
            controlPoints.Remove(point);
        }

        public override Point CalculatePoint(double splineProgress)
        {
            if (splineProgress == 1)
            {
                return new Point(controlPoints[^1].Y, controlPoints[^1].Y); // Return a copy of the last point if splineProgress is 1.
            }

            double timeValue = splineProgress * controlPoints.Count;
            int p1Index = (int)timeValue;
            double interpolationTValue = timeValue % 1;

            Point interpolatedPoint = new(0, 0);

            int p0Index = WrappingIndexAdd(p1Index, -1);
            int p2Index = WrappingIndexAdd(p1Index, 1);
            int p3Index = WrappingIndexAdd(p1Index, 2);

            interpolatedPoint.X = Interpolate(interpolationTValue, controlPoints[p0Index].X, controlPoints[p1Index].X, controlPoints[p2Index].X, controlPoints[p2Index].X);
            interpolatedPoint.Y = Interpolate(interpolationTValue, controlPoints[p0Index].Y, controlPoints[p1Index].Y, controlPoints[p2Index].Y, controlPoints[p2Index].Y);

            return interpolatedPoint;
        }
    }
}
