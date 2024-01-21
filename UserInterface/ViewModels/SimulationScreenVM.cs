using System;
using System.ComponentModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Navigation;
using UserInterface.HelperClasses;

namespace UserInterface.ViewModels
{
    public class SimulationScreenVM : ViewModel
    {
        #region Fields, Properties and Enums
        private SidePanelButton? currentButton;

        private float visLowerBound;
        private float visUpperBound;

        private float[] velocity;
        private float[] pressure;
        private float[] streamFunction;
        private FieldParameters pressureParameters;
        private FieldParameters velocityParameters;
        private SelectedField selectedField;

        private BackendManager backendManager;
        private CancellationTokenSource backendCTS;
        private VisualisationControl visualisationControl;
        private MovingAverage<float> visFPSAverage;
        private MovingAverage<float> backFPSAverage;

        private int dataWidth;
        private int dataHeight;


        public string? CurrentButton //Conversion between string and internal enum value done in property
        {
            get
            {
                if (currentButton == null) return null;
                return Enum.GetName(typeof(SidePanelButton), currentButton);
            }
            set
            {
                if (value == null)
                {
                    currentButton = null;
                }
                else
                {
                    currentButton = (SidePanelButton)Enum.Parse(typeof(SidePanelButton), value);
                }
                OnPropertyChanged(this, nameof(currentButton));
            }
        }

        // Option for Properties that are in ParameterHolder:
        // Property is just an accessor for the ParameterHolder property.
        // Needs to then update with NotifyPropertyChanged.

        public float InVel
        {
            get => parameterHolder.InflowVelocity.Value;
            set
            {
                parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, value);
                OnPropertyChanged(this, nameof(InVel));
            }
        }

        public float Chi
        {
            get => parameterHolder.SurfaceFriction.Value;
            set
            {
                parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, value);
                OnPropertyChanged(this, nameof(Chi));
            }
        }

        public float VisMin
        {
            get => parameterHolder.FieldParameters.Value.min;
            set
            {
                parameterHolder.FieldParameters = ModifyParameterValue(parameterHolder.FieldParameters, ModifyFieldParameters(parameterHolder.FieldParameters.Value, null, value, null));
                OnPropertyChanged(this, nameof(VisMin));
            }
        }

        public float VisMax
        {
            get => parameterHolder.FieldParameters.Value.max;
            set
            {
                parameterHolder.FieldParameters = ModifyParameterValue(parameterHolder.FieldParameters, ModifyFieldParameters(parameterHolder.FieldParameters.Value, null, null, value));
                OnPropertyChanged(this, nameof(VisMax));
            }
        }

        public float VisLowerBound
        {
            get => visLowerBound;
            private set
            {
                visLowerBound = value;
                OnPropertyChanged(this, nameof(VisLowerBound));
            }
        }

        public float VisUpperBound
        {
            get => visUpperBound;
            private set
            {
                visUpperBound = value;
                OnPropertyChanged(this, nameof(VisUpperBound));
            }
        }

        public float ContourSpacing
        {
            get => parameterHolder.ContourSpacing.Value;
            set
            {
                parameterHolder.ContourSpacing = ModifyParameterValue(parameterHolder.ContourSpacing, value);
                OnPropertyChanged(this, nameof(ContourSpacing));
            }
        }

        public float ContourTolerance
        {
            get => parameterHolder.ContourTolerance.Value;
            set
            {
                parameterHolder.ContourTolerance = ModifyParameterValue(parameterHolder.ContourTolerance, value);
                OnPropertyChanged(this, nameof(ContourTolerance));
            }
        }

        public bool PressureChecked
        {
            get { return selectedField == SelectedField.Pressure; }
            set
            {
                if (value)
                {
                    selectedField = SelectedField.Pressure;
                }
                else
                {
                    selectedField = SelectedField.Velocity;
                }

                OnPropertyChanged(this, nameof(PressureChecked));
                SwitchFieldParameters();
            }
        }

        public bool VelocityChecked
        {
            get { return selectedField == SelectedField.Velocity; }
            set
            {
                if (value) selectedField = SelectedField.Velocity;
                else selectedField = SelectedField.Pressure;
                OnPropertyChanged(this, nameof(VelocityChecked));
                SwitchFieldParameters();
            }
        }

        public bool StreamlinesEnabled
        {
            get => parameterHolder.DrawContours.Value;
            set
            {
                parameterHolder.DrawContours = ModifyParameterValue(parameterHolder.DrawContours, value);
                OnPropertyChanged(this, nameof(StreamlinesEnabled));
            }
        }

        public VisualisationControl VisualisationControl { get => visualisationControl; }

        public float VisFPS { get => visFPSAverage.Average; }

        public float BackFPS { get => backFPSAverage.Average; }

        public Commands.SwitchPanel SwitchPanelCommand { get; set; }
        public Commands.StopBackend StopBackendCommand { get; set; }
        public CancellationTokenSource BackendCTS { get => backendCTS; set => backendCTS = value; }

        private enum SidePanelButton //Different side panels on SimluationScreen
        {
            BtnParametersSelect,
            BtnUnitsSelect,
            BtnVisualisationSettingsSelect,
            BtnRecordingSelect
        }
        private enum SelectedField
        {
            Pressure,
            Velocity
        }
        #endregion

        public event CancelEventHandler StopBackendExecuting;

        public SimulationScreenVM(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            #region Parameters related to View
            currentButton = null; // Initially no panel selected
            InVel = parameterHolder.InflowVelocity.Value;
            Chi = parameterHolder.SurfaceFriction.Value;

            SwitchPanelCommand = new Commands.SwitchPanel(this);
            StopBackendCommand = new Commands.StopBackend(this);
            #endregion

            #region Parameters related to Backend
            backendCTS = new CancellationTokenSource();
            StopBackendExecuting += (object? sender, CancelEventArgs e) => backendCTS.Cancel();

            backendManager = new BackendManager(parameterHolder);
            bool connectionSuccess = backendManager.ConnectBackend();
            if (!connectionSuccess)
            {
                MessageBox.Show("Fatal error: backend did not connect properly.");
                throw new IOException("Backend did not connect properly.");
            }

            backendManager.SendAllParameters();
            velocity = new float[backendManager.FieldLength];
            pressure = new float[backendManager.FieldLength];
            streamFunction = new float[backendManager.FieldLength];
            dataWidth = backendManager.IMax;
            dataHeight = backendManager.JMax;
            SendObstacles();

            Task.Run(StartComputation);
            backFPSAverage = new MovingAverage<float>(DefaultParameters.FPS_WINDOW_SIZE);
            backendManager.PropertyChanged += BackFPSUpdate;
            #endregion

            #region Parameters related to Visualisation
            SetFieldDefaults();
            selectedField = SelectedField.Velocity; // Velocity selected initially.
            SwitchFieldParameters();

            visualisationControl = new VisualisationControl(parameterHolder, streamFunction, dataWidth, dataHeight); // Content of VisualisationControlHolder is bound to this.
            visualisationControl.PropertyChanged += VisFPSUpdate; // FPS updating method
            visFPSAverage = new MovingAverage<float>(DefaultParameters.FPS_WINDOW_SIZE);
            #endregion
        }

        private void SetFieldDefaults()
        {
            velocityParameters.field = velocity;
            velocityParameters.min = DefaultParameters.VELOCITY_MIN;
            velocityParameters.max = DefaultParameters.VELOCITY_MAX;
            pressureParameters.field = pressure;
            pressureParameters.min = DefaultParameters.PRESSURE_MIN;
            pressureParameters.max = DefaultParameters.PRESSURE_MAX;
        }

        private void SwitchFieldParameters()
        {
            if (selectedField == SelectedField.Pressure)
            {
                parameterHolder.FieldParameters = ModifyParameterValue(parameterHolder.FieldParameters, pressureParameters);
                VisLowerBound = DefaultParameters.PRESSURE_MIN;
                VisUpperBound = DefaultParameters.PRESSURE_MAX;
            }
            else // Velocity selected
            {
                parameterHolder.FieldParameters = ModifyParameterValue(parameterHolder.FieldParameters, velocityParameters);
                VisLowerBound = DefaultParameters.VELOCITY_MIN;
                VisUpperBound = DefaultParameters.VELOCITY_MAX;
            }
            VisMin = parameterHolder.FieldParameters.Value.min;
            VisMax = parameterHolder.FieldParameters.Value.max;


        }

        private FieldParameters ModifyFieldParameters(FieldParameters fieldParameters, float[]? newField, float? newMin, float? newMax)
        {
            if (newField is not null)
            {
                fieldParameters.field = newField;
            }
            if (newMin is not null)
            {
                fieldParameters.min = (float)newMin;
            }
            if (newMax is not null)
            {
                fieldParameters.max = (float)newMax;
            }
            return fieldParameters;
        }

        private bool SendObstacles() // Temporary method to create a square of obstacle cells
        {
            bool[] obstacles = new bool[(dataWidth + 2) * (dataHeight + 2)];

            for (int i = 1; i <= dataWidth; i++)
            {
                for (int j = 1; j <= dataHeight; j++)
                {
                    obstacles[i * (dataHeight + 2) + j] = true; // Set cells to fluid
                }
            }

            int boundaryLeft = (int)(0.15 * dataWidth);
            int boundaryRight = (int)(0.25 * dataWidth);
            int boundaryBottom = (int)(0.45 * dataHeight);
            int boundaryTop = (int)(0.55 * dataHeight);

            for (int i = boundaryLeft; i < boundaryRight; i++)
            { // Create a square of boundary cells
                for (int j = boundaryBottom; j < boundaryTop; j++)
                {
                    obstacles[i * (dataHeight + 2) + j] = false;
                }
            }


            return backendManager.SendObstacles(obstacles);
        }

        private void StartComputation()
        {
            try
            {
                backendManager.GetFieldStreamsAsync(velocity, null, pressure, streamFunction, backendCTS.Token);
            }
            catch (IOException e)
            {
                MessageBox.Show(e.Message);
            }
            catch (Exception e)
            {
                MessageBox.Show($"Generic error: {e.Message}");
            }
        }

        private void VisFPSUpdate(object? sender, PropertyChangedEventArgs e)
        {
            visFPSAverage.UpdateAverage(visualisationControl.FramesPerSecond);
            OnPropertyChanged(this, nameof(VisFPS));
        }

        private void BackFPSUpdate(object? sender, PropertyChangedEventArgs e)
        {
            backFPSAverage.UpdateAverage(backendManager.FramesPerSecond);
            OnPropertyChanged(this, nameof(BackFPS));
        }
    }
}
