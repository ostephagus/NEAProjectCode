using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace UserInterface.HelperClasses
{
    public enum ParameterUsage
    {
        Backend,
        Visualisation
    }


    public struct ParameterStruct<T>
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

    public class ParameterHolder : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        // Backend parameters
        private ParameterStruct<float> width;
        private ParameterStruct<float> height;
        private ParameterStruct<float> timeStepSafetyFactor;
        private ParameterStruct<float> relaxationParameter;
        private ParameterStruct<float> pressureResidualTolerance;
        private ParameterStruct<float> pressureMaxIterations;
        private ParameterStruct<float> reynoldsNumber;
        private ParameterStruct<float> fluidViscosity;
        private ParameterStruct<float> fluidVelocity;
        private ParameterStruct<float> fluidDensity;
        private ParameterStruct<float> surfaceFriction;

        // Visualisation parameters
        private ParameterStruct<FieldParameters> fieldParameters;
        private ParameterStruct<bool> drawContours;
        private ParameterStruct<float> contourTolerance;
        private ParameterStruct<float> contourSpacing;

        #region Properties
        public ParameterStruct<float> Width
        {
            get => width;

            set
            {
                width = value;
                OnPropertyChanged(width.Value);
            }
        }
        public ParameterStruct<float> Height
        {
            get => height;

            set
            {
                height = value;
                OnPropertyChanged(height.Value);
            }
        }
        public ParameterStruct<float> TimeStepSafetyFactor
        {
            get => timeStepSafetyFactor;

            set
            {
                timeStepSafetyFactor = value;
                OnPropertyChanged(TimeStepSafetyFactor.Value);
            }
        }
        public ParameterStruct<float> RelaxationParameter
        {
            get => relaxationParameter;

            set
            {
                relaxationParameter = value;
                OnPropertyChanged(relaxationParameter.Value);
            }
        }
        public ParameterStruct<float> PressureResidualTolerance
        {
            get => pressureResidualTolerance;

            set
            {
                pressureResidualTolerance = value;
                OnPropertyChanged(pressureResidualTolerance.Value);
            }
        }
        public ParameterStruct<float> PressureMaxIterations
        {
            get => pressureMaxIterations;

            set
            {
                pressureMaxIterations = value;
                OnPropertyChanged(pressureMaxIterations.Value);
            }
        }
        public ParameterStruct<float> ReynoldsNumber
        {
            get => reynoldsNumber;

            set
            {
                reynoldsNumber = value;
                OnPropertyChanged(reynoldsNumber.Value);
            }
        }

        public ParameterStruct<float> FluidViscosity
        {
            get => fluidViscosity;
            set
            {
                fluidViscosity = value;
                OnPropertyChanged(fluidViscosity.Value);
            }
        }

        public ParameterStruct<float> InflowVelocity
        {
            get => fluidVelocity;

            set
            {
                fluidVelocity = value;
                OnPropertyChanged(fluidVelocity.Value);
            }
        }

        public ParameterStruct<float> FluidDensity
        {
            get => fluidDensity;
            set
            {
                fluidDensity = value;
                OnPropertyChanged(FluidDensity.Value);
            }
        }

        public ParameterStruct<float> SurfaceFriction
        {
            get => surfaceFriction;

            set
            {
                surfaceFriction = value;
                OnPropertyChanged(surfaceFriction.Value);
            }
        }
        public ParameterStruct<FieldParameters> FieldParameters
        {
            get => fieldParameters;

            set
            {
                fieldParameters = value;
            }
        }
        public ParameterStruct<float> ContourTolerance
        {
            get => contourTolerance;

            set
            {
                contourTolerance = value;
            }
        }
        public ParameterStruct<float> ContourSpacing
        {
            get => contourSpacing;

            set
            {
                contourSpacing = value;
            }
        }
        public ParameterStruct<bool> DrawContours
        {
            get => drawContours;

            set
            {
                drawContours = value;
            }
        }
        #endregion

        public ParameterHolder(float width, float height, float timeStepSafetyFactor, float relaxationParameter, float pressureResidualTolerance, float pressureMaxIterations, float reynoldsNumber, float fluidViscosity, float fluidVelocity, float fluidDensity, float surfaceFriction, FieldParameters fieldParameters, bool drawContours, float contourTolerance, float contourSpacing)
        {

            this.width = new ParameterStruct<float>(width, ParameterUsage.Backend, false);
            this.height = new ParameterStruct<float>(height, ParameterUsage.Backend, false);
            this.timeStepSafetyFactor = new ParameterStruct<float>(timeStepSafetyFactor, ParameterUsage.Backend, true);
            this.relaxationParameter = new ParameterStruct<float>(relaxationParameter, ParameterUsage.Backend, false);
            this.pressureResidualTolerance = new ParameterStruct<float>(pressureResidualTolerance, ParameterUsage.Backend, true);
            this.pressureMaxIterations = new ParameterStruct<float>(pressureMaxIterations, ParameterUsage.Backend, true);
            this.reynoldsNumber = new ParameterStruct<float>(reynoldsNumber, ParameterUsage.Backend, false);
            this.fluidViscosity = new ParameterStruct<float>(fluidViscosity, ParameterUsage.Backend, false);
            this.fluidVelocity = new ParameterStruct<float>(fluidVelocity, ParameterUsage.Backend, true);
            this.fluidDensity = new ParameterStruct<float>(fluidDensity, ParameterUsage.Backend, false);
            this.surfaceFriction = new ParameterStruct<float>(surfaceFriction, ParameterUsage.Backend, true);
            this.fieldParameters = new ParameterStruct<FieldParameters>(fieldParameters, ParameterUsage.Visualisation, true);
            this.drawContours = new ParameterStruct<bool>(drawContours, ParameterUsage.Visualisation, true);
            this.contourTolerance = new ParameterStruct<float>(contourTolerance, ParameterUsage.Visualisation, true);
            this.contourSpacing = new ParameterStruct<float>(contourSpacing, ParameterUsage.Visualisation, true);
        }

        private void OnPropertyChanged(float value, [CallerMemberName] string name = "")
        {
            PropertyChanged?.Invoke(this, new ParameterChangedEventArgs(name, value));
        }

        public void ReadParameters(string fileName)
        {
            throw new NotImplementedException("ReadParameters not yet implemented");
        }
    }
}
