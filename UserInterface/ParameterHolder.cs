using System;

namespace UserInterface
{
    public enum ParameterUsage
    {
        Backend,
        Visualisation
    }

    public class ParameterStruct<T>
    {
        /// <summary>
        /// Initialises a parameter struct with a default value separate to its initial value.
        /// </summary>
        /// <param name="defaultValue">The default value for the parameter.</param>
        /// <param name="value">The initial value for the paramter.</param>
        /// <param name="usage">Where in the program the parameter is used.</param>
        /// <param name="canChangeWhileRunning">A <c>bool</c> to indicate whether the parameter can change while the simulation is running.</param>
        public ParameterStruct(T defaultValue, T value, ParameterUsage usage, bool canChangeWhileRunning)
        {
            DefaultValue = defaultValue;
            Value = value;
            Usage = usage;
            CanChangeWhileRunning = canChangeWhileRunning;
        }

        /// <summary>
        /// Initialises a parameter struct with its default value.
        /// </summary>
        /// <param name="value">The default value for the parameter, to be used as its initial value also.</param>
        /// <param name="usage">Where in the program the parameter is used.</param>
        /// <param name="canChangeWhileRunning">A <c>bool</c> to indicate whether the parameter can change while the simulation is running.</param>
        public ParameterStruct(T value, ParameterUsage usage, bool canChangeWhileRunning)
        {
            DefaultValue = value;
            Value = DefaultValue;
            Usage = usage;
            CanChangeWhileRunning = canChangeWhileRunning;
        }

        public T DefaultValue { get; }
        public T Value { get; set; }
        public ParameterUsage Usage { get; }
        public bool CanChangeWhileRunning { get; }

        public void Reset()
        {
            Value = DefaultValue;
        }
    }

    public struct FieldParameters
    {
        public float[] field;
        public float min;
        public float max;
    }

    public class ParameterHolder
    {
        // Backend parameters
        private ParameterStruct<float> width;
        private ParameterStruct<float> height;
        private ParameterStruct<float> timeStepSafetyFactor;
        private ParameterStruct<float> relaxationParameter;
        private ParameterStruct<float> pressureResidualTolerance;
        private ParameterStruct<float> pressureMaxIterations;
        private ParameterStruct<float> reynoldsNumber;
        private ParameterStruct<float> fluidVelocity;
        private ParameterStruct<float> surfaceFriction;

        // Visualisation parameters
        private ParameterStruct<FieldParameters> fieldParameters;
        private ParameterStruct<float> contourTolerance;
        private ParameterStruct<float> contourSpacing;

        public ParameterStruct<float> Width { get => width; set => width = value; }
        public ParameterStruct<float> Height { get => height; set => height = value; }
        public ParameterStruct<float> TimeStepSafetyFactor { get => timeStepSafetyFactor; set => timeStepSafetyFactor = value; }
        public ParameterStruct<float> RelaxationParameter { get => relaxationParameter; set => relaxationParameter = value; }
        public ParameterStruct<float> PressureResidualTolerance { get => pressureResidualTolerance; set => pressureResidualTolerance = value; }
        public ParameterStruct<float> PressureMaxIterations { get => pressureMaxIterations; set => pressureMaxIterations = value; }
        public ParameterStruct<float> ReynoldsNumber { get => reynoldsNumber; set => reynoldsNumber = value; }
        public ParameterStruct<float> FluidVelocity { get => fluidVelocity; set => fluidVelocity = value; }
        public ParameterStruct<float> SurfaceFriction { get => surfaceFriction; set => surfaceFriction = value; }
        public ParameterStruct<FieldParameters> FieldParameters { get => fieldParameters; set => fieldParameters = value; }
        public ParameterStruct<float> ContourTolerance { get => contourTolerance; set => contourTolerance = value; }
        public ParameterStruct<float> ContourSpacing { get => contourSpacing; set => contourSpacing = value; }

        public ParameterHolder(float width, float height, float timeStepSafetyFactor, float relaxationParameter, float pressureResidualTolerance, float pressureMaxIterations, float reynoldsNumber, float fluidVelocity, float surfaceFriction, FieldParameters fieldParameters, float contourTolerance, float contourSpacing)
        {

            this.width =                     new ParameterStruct<float>(width, ParameterUsage.Backend, false);
            this.height =                    new ParameterStruct<float>(height, ParameterUsage.Backend, false);
            this.timeStepSafetyFactor =      new ParameterStruct<float>(timeStepSafetyFactor, ParameterUsage.Backend, false);
            this.relaxationParameter =       new ParameterStruct<float>(relaxationParameter, ParameterUsage.Backend, false);
            this.pressureResidualTolerance = new ParameterStruct<float>(pressureResidualTolerance, ParameterUsage.Backend, false);
            this.pressureMaxIterations =     new ParameterStruct<float>(pressureMaxIterations, ParameterUsage.Backend, false);
            this.reynoldsNumber =            new ParameterStruct<float>(reynoldsNumber, ParameterUsage.Backend, false);
            this.fluidVelocity =             new ParameterStruct<float>(fluidVelocity, ParameterUsage.Backend, true);
            this.surfaceFriction =           new ParameterStruct<float>(surfaceFriction, ParameterUsage.Backend, false);
            this.fieldParameters =           new ParameterStruct<FieldParameters>(fieldParameters, ParameterUsage.Visualisation, true);
            this.contourTolerance =          new ParameterStruct<float>(contourTolerance, ParameterUsage.Visualisation, true);
            this.contourSpacing =            new ParameterStruct<float>(width, ParameterUsage.Visualisation, false);
        }

        public void ReadParameters(string fileName)
        {
            throw new NotImplementedException("ReadParameters not yet implemented");
        }
    }
}
