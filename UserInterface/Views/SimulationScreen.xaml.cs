using System;
using System.Diagnostics;
using System.ComponentModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using UserInterface.HelperClasses;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : SwappableScreen, INotifyPropertyChanged
    {
        #region Fields and properties
        private SidePanelButton? currentButton;
        private readonly CancellationTokenSource backendCancellationTokenSource;

        private BackendManager? backendManager;
        private VisualisationControl? visualisationControl;
        private MovingAverage<float> visFPSAverage;
        private const int VIS_FPS_WINDOW_SIZE = 500;

        private float[]? horizontalVelocity;
        private float[]? pressure;
        private float[]? streamFunction;
        private int dataWidth;
        private int dataHeight;

        private const int min = -1;
        private const int max = 2;

        public event PropertyChangedEventHandler? PropertyChanged;
        public static event CancelEventHandler? StopBackendExecuting;

        public string VisFPS { get; set; }
        public string BackFPS { get; set; } = "0";

        public ICommand Command_StopBackendExecuting { get; } = new Commands.StopBackend();

        public string? CurrentButton //Conversion between string and internal enum value done in property
        {
            get {
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
                PropertyChanged.Invoke(this, new PropertyChangedEventArgs(nameof(currentButton)));
            }
        }

        public enum SidePanelButton //Different side panels on SimluationScreen
        {
            BtnParametersSelect,
            BtnUnitsSelect,
            BtnVisualisationSettingsSelect,
            BtnRecordingSelect
        }
        #endregion

        #region Constructors and methods
        public SimulationScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            DataContext = this;
            currentButton = null;
            backendCancellationTokenSource = new CancellationTokenSource();
            visFPSAverage = new(VIS_FPS_WINDOW_SIZE);
            StopBackendExecuting += (object? sender, CancelEventArgs e) => backendCancellationTokenSource.Cancel();
            if (StartComponents())
            {
                Task.Run(StartComputation); // Asynchronously run the computation
                SetSliders();
            }
        }

        private bool StartComponents()
        {
            backendManager = new BackendManager(parameterHolder);
            bool success = backendManager.ConnectBackend();
            if (!success)
            {
                MessageBox.Show("Fatal error: backend did not connect properly.");
                return false;
            }

            backendManager.SendAllParameters();
            bool[] obstacles = new bool[(backendManager.IMax + 2) * (backendManager.JMax + 2)];
            
            for (int i = 1; i <= backendManager.IMax; i++)
            {
                for (int j = 1; j <= backendManager.JMax; j++)
                {
                    obstacles[i * (backendManager.JMax + 2) + j] = true; // Set cells to fluid
                }
            }

            int boundaryLeft = (int)(0.15 * backendManager.IMax);
            int boundaryRight = (int)(0.25 * backendManager.IMax);
            int boundaryBottom = (int)(0.45 * backendManager.JMax);
            int boundaryTop = (int)(0.55 * backendManager.JMax);

            for (int i = boundaryLeft; i < boundaryRight; i++)
            { // Create a square of boundary cells
                for (int j = boundaryBottom; j < boundaryTop; j++)
                {
                    obstacles[i * (backendManager.JMax + 2) + j] = false;
                }
            }

            Trace.WriteLine(backendManager.SendObstacles(obstacles) ? "Obstacle send successful" : "obstacle send unsuccessful");

            horizontalVelocity = new float[backendManager.FieldLength];
            pressure = new float[backendManager.FieldLength];
            streamFunction = new float[backendManager.FieldLength];
            dataWidth = backendManager.IMax;
            dataHeight = backendManager.JMax;

            FieldParameters fieldParameters = parameterHolder.FieldParameters.Value; // Get the field parameters struct (structs must be copied, edited, then saved)
            fieldParameters.field = horizontalVelocity; // Set default field to be horizontal velocity
            fieldParameters.min = min; // Set the defaults to constants min...
            fieldParameters.max = max; // ...and max
            parameterHolder.FieldParameters = ModifyParameterValue(parameterHolder.FieldParameters, fieldParameters); // Save it to the parameter holder

            visualisationControl = new VisualisationControl(parameterHolder, streamFunction, dataWidth, dataHeight);
            VisualisationControlHolder.Content = visualisationControl;
            visualisationControl.PropertyChanged += VisFPSUpdate;
            return true;
        }

        private void StartComputation()
        {
            try
            {
                backendManager.GetFieldStreamsAsync(horizontalVelocity, null, pressure, streamFunction, backendCancellationTokenSource.Token);
            } catch (IOException e)
            {
                MessageBox.Show(e.Message);
            } catch (Exception e)
            {
                MessageBox.Show($"Generic error: {e.Message}");
            }
        }

        private void SetSliders()
        {
            SliderInVel.Value = parameterHolder.InflowVelocity.Value;
            SliderChi.Value = parameterHolder.SurfaceFriction.Value;

            FieldParameters fieldParameters = parameterHolder.FieldParameters.Value;
            RBPressure.IsChecked = fieldParameters.field.Equals(pressure);
            RBVelocity.IsChecked = fieldParameters.field.Equals(horizontalVelocity);
            SliderMin.Value = fieldParameters.min;
            SliderMax.Value = fieldParameters.max;

            SliderContourSpacing.Value = parameterHolder.ContourSpacing.Value;
            SliderContourTolerance.Value = parameterHolder.ContourTolerance.Value;
        }
        #endregion

        #region Event handlers
        private void PanelButton_Click(object sender, RoutedEventArgs e)
        {
            string name = ((FrameworkElement)sender).Name;
            if (name == CurrentButton) //If the button of the currently open panel is clicked, close all panels (null)
            {
                CurrentButton = null;
            }
            else
            {
                CurrentButton = name; //If any other panel is open, or no panel is open, open the one corresponding to the button.
            }
        }

        public static void RaiseStopBackendExecuting()
        {
            StopBackendExecuting.Invoke(null, new CancelEventArgs());
        }

        private void SliderInVel_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, (float)SliderInVel.Value);
        }

        private void SliderChi_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, (float)SliderChi.Value);
        }

        private void SliderContourTolerance_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.ContourTolerance = ModifyParameterValue(parameterHolder.ContourTolerance, (float)SliderContourTolerance.Value);
        }

        private void SliderContourSpacing_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.ContourSpacing = ModifyParameterValue(parameterHolder.ContourSpacing, (float)SliderContourSpacing.Value);
        }

        private void BtnFieldParamsSave_Click(object sender, RoutedEventArgs e)
        {
            FieldParameters fieldParameters = parameterHolder.FieldParameters.Value;
            fieldParameters.field = (RBPressure.IsChecked ?? false) ? pressure : horizontalVelocity; // Set the field according to whether the pressure is checked
            fieldParameters.min = (float)SliderMin.Value;
            fieldParameters.max = (float)SliderMax.Value;
            parameterHolder.FieldParameters = ModifyParameterValue(parameterHolder.FieldParameters, fieldParameters);
        }

        private void CBContourLines_Click(object sender, RoutedEventArgs e)
        {
            parameterHolder.DrawContours = ModifyParameterValue(parameterHolder.DrawContours, CBContourLines.IsChecked ?? false);
        }

        private void VisFPSUpdate(object? sender, PropertyChangedEventArgs e)
        {
            float newAverage = visFPSAverage.UpdateAverage(visualisationControl.FramesPerSecond); // Update the average and get the new average returned
            RunVisFPS.Text = newAverage.ToString("0"); // Format the new average and display it
        }
        #endregion
    }
}
