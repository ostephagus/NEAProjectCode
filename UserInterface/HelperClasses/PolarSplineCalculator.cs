using System;
using System.Collections.Generic;
using System.Windows;
using MathNet.Numerics.LinearAlgebra;

namespace UserInterface.HelperClasses
{
    public class PolarSplineCalculator : SplineCalculator
    {
        private Converters.RectangularToPolar recToPolConverter;

        private List<PolarPoint> controlPoints;

        private Vector<double>? splineFunctionCoefficients;
        private bool isValidSpline;

        public override bool IsValidSpline { get => isValidSpline; protected set => isValidSpline = value; }

        public PolarSplineCalculator()
        {
            controlPoints = new List<PolarPoint>();
            IsValidSpline = false;
            recToPolConverter = new Converters.RectangularToPolar();
        }

        /// <summary>
        /// Adds one to <paramref name="input"/>. If that is equal to <paramref name="comparison"/>, return 0.
        /// </summary>
        /// <param name="input">The input</param>
        /// <param name="comparison">The number that <paramref name="input"/> must be less than.</param>
        /// <returns><paramref name="input"/> + 1, or 0.</returns>
        private static int WrapAdd(int input, int comparison) => (input + 1) == comparison ? 0 : (input + 1);

        private PolarPoint QuickConvert(Point point) => (PolarPoint)recToPolConverter.Convert(point, typeof(PolarPoint), null, System.Globalization.CultureInfo.CurrentCulture);
        private Point QuickConvert(PolarPoint point) => (Point)recToPolConverter.ConvertBack(point, typeof(Point), null, System.Globalization.CultureInfo.CurrentCulture);

        private void CalculateSplineFunction()
        {
            int numSegments = controlPoints.Count; // n segments for n points, beacuse the final segment wraps back around to the first.
            // For each segment:
            // Eq0: passes ith control point.
            // Eq1: passes through (i + 1)th control point.
            // Eq2: Derivative of ith segment at (i + 1)th x coordinate is equal to (i + 1)th segment at that x coordinate
            // Eq3: As above but for second derivative.
            // Form of each cubic: ax^3 + bx^2 + cx + d.
            Matrix<double> cubicCoefficients = Matrix<double>.Build.Dense(4 * numSegments, 4 * numSegments); // Create a new matrix for coefficients with size 4 * numSegments, because there are 4 equations and 4 coefficients (it's a cubic) per segment.
            Vector<double> rhsValues = Vector<double>.Build.Dense(4 * numSegments);
            for (int segmentNo = 0; segmentNo < numSegments; segmentNo++)
            {
                PolarPoint startPoint = controlPoints[segmentNo];
                PolarPoint endPoint;

                if (segmentNo < numSegments - 1)
                {
                    endPoint = controlPoints[segmentNo + 1];
                }
                else // Last segment needs to wrap around and add 2 pi.
                {
                    endPoint = new PolarPoint(controlPoints[0].Radius, controlPoints[0].Angle + 2 * Math.PI);
                }

                // Eq0: substitute angle of start point into cubic and equate it to radius of point.
                cubicCoefficients[segmentNo * 4 + 0, segmentNo * 4 + 0] = Math.Pow(startPoint.Angle, 3);
                cubicCoefficients[segmentNo * 4 + 0, segmentNo * 4 + 1] = Math.Pow(startPoint.Angle, 2);
                cubicCoefficients[segmentNo * 4 + 0, segmentNo * 4 + 2] = startPoint.Angle;
                cubicCoefficients[segmentNo * 4 + 0, segmentNo * 4 + 3] = 1;
                rhsValues[segmentNo * 4 + 0] = startPoint.Radius;

                // Eq1: substitute angle of end point into cubic.
                cubicCoefficients[segmentNo * 4 + 1, segmentNo * 4 + 0] = Math.Pow(endPoint.Angle, 3);
                cubicCoefficients[segmentNo * 4 + 1, segmentNo * 4 + 1] = Math.Pow(endPoint.Angle, 2);
                cubicCoefficients[segmentNo * 4 + 1, segmentNo * 4 + 2] = endPoint.Angle;
                cubicCoefficients[segmentNo * 4 + 1, segmentNo * 4 + 3] = 1;
                rhsValues[segmentNo * 4 + 1] = endPoint.Radius;

                // Eq2: derivatives match at end point
                cubicCoefficients[segmentNo * 4 + 2, segmentNo * 4 + 0] = 3 * Math.Pow(endPoint.Angle, 2);
                cubicCoefficients[segmentNo * 4 + 2, segmentNo * 4 + 1] = 2 * endPoint.Angle;
                cubicCoefficients[segmentNo * 4 + 2, segmentNo * 4 + 2] = 1;
                cubicCoefficients[segmentNo * 4 + 2, WrapAdd(segmentNo, numSegments) * 4 + 0] = -3 * Math.Pow(endPoint.Angle, 2);
                cubicCoefficients[segmentNo * 4 + 2, WrapAdd(segmentNo, numSegments) * 4 + 1] = -2 * endPoint.Angle;
                cubicCoefficients[segmentNo * 4 + 2, WrapAdd(segmentNo, numSegments) * 4 + 2] = -1;
                // RHS is 0.

                // Eq3: second derivatives match at end point
                cubicCoefficients[segmentNo * 4 + 3, segmentNo * 4 + 0] = 6 * endPoint.Angle;
                cubicCoefficients[segmentNo * 4 + 3, segmentNo * 4 + 1] = 2;
                cubicCoefficients[segmentNo * 4 + 3, WrapAdd(segmentNo, numSegments) * 4 + 0] = -6 * endPoint.Angle;
                cubicCoefficients[segmentNo * 4 + 3, WrapAdd(segmentNo, numSegments) * 4 + 1] = -2;
                // RHS is 0.
            }

            splineFunctionCoefficients = cubicCoefficients.Solve(rhsValues);
        }

        public override void AddControlPoint(Point point)
        {
            controlPoints.Add(QuickConvert(point));
            controlPoints.Sort();
            if (controlPoints.Count >= 3)
            {
                IsValidSpline = true;
                CalculateSplineFunction();
            }
        }

        public override void ModifyControlPoint(Point oldPoint, Point newPoint)
        {
            controlPoints.Remove(QuickConvert(oldPoint));
            controlPoints.Add(QuickConvert(newPoint));
            controlPoints.Sort();
            if (controlPoints.Count >= 3)
            {
                CalculateSplineFunction();
            }
        }

        public override void RemoveControlPoint(Point point)
        {
            if (controlPoints.Count < 3)
            {
                throw new InvalidOperationException("A spline must have at least 3 points.");
            }
            controlPoints.Remove(QuickConvert(point));
            CalculateSplineFunction();
        }

        public override Point CalculatePoint(double splineProgress)
        {
            if (splineFunctionCoefficients is null)
            {
                throw new InvalidOperationException("CalculatePoint cannot be called when there are fewer than 3 coordinates supplied.");
            }
            if (splineProgress < 0 || splineProgress > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(splineProgress), "Supplied spline progress must be between 0 and 1.");
            }

            double theta = 2 * Math.PI * splineProgress;
            if (theta < controlPoints[0].Angle) theta += 2 * Math.PI; // If theta is before the first control point, it is in the last segment so add 2 pi to it so it conforms to the bounds of the last segment.

            int segmentNo = controlPoints.Count - 1; // If theta is less than none of the coordinates then it must be in the last segment. segmentNo starts as this in case none of the conditions in the loop are met.

            for (int i = 0; i < controlPoints.Count - 1; i++)
            {
                if (theta < controlPoints[i + 1].Angle)
                {
                    segmentNo = i;
                    break;
                }
            }
            double radius = splineFunctionCoefficients[4 * segmentNo + 0] * Math.Pow(theta, 3)
            + splineFunctionCoefficients[4 * segmentNo + 1] * Math.Pow(theta, 2)
            + splineFunctionCoefficients[4 * segmentNo + 2] * theta
            + splineFunctionCoefficients[4 * segmentNo + 3];

            PolarPoint finalPoint;

            if (radius > 0)
            {
                finalPoint = new PolarPoint(radius, theta);
            }
            else
            {
                IsValidSpline = false;
                finalPoint = new PolarPoint(0, theta);
            }

            return QuickConvert(finalPoint);
        }
    }
}
