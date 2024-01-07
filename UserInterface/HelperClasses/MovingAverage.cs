using System;
using System.Numerics;
namespace UserInterface.HelperClasses
{
    public class MovingAverage<T> where T : INumber<T>
    {
        private CircularQueue<T> dataPoints;
        private readonly int windowSize;

        private T currentSum = default; // Contains the sum of all the current data points

        public T Average { get; private set; }

        public MovingAverage(int windowSize)
        {
            this.windowSize = windowSize;
            dataPoints = new CircularQueue<T>(windowSize);
        }

        public T UpdateAverage(T newValue)
        {
            if (dataPoints.Count == windowSize)
            {
                currentSum -= dataPoints.Dequeue(); // Take the first item off the sum (discarding it)
            }

            currentSum += newValue;
            dataPoints.Enqueue(newValue);

            Average = currentSum / (T)Convert.ChangeType(dataPoints.Count, typeof(T)); // Divide the current sum by the number of data points. Conversion between int and generic T had to use ChangeType
            return Average;
        }
    }
}
