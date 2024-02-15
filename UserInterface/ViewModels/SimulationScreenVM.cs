using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using UserInterface.Converters;
using UserInterface.HelperClasses;
using UserInterface.HelperControls;

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
        private UnitConversionPanel unitsPanel;
        private VisualisationControl visualisationControl;
        private MovingAverage<float> visFrameTimeAverage;
        private MovingAverage<float> backFrameTimeAverage;
        private MovingAverage<float> dragCoefficientAverage;
        private RectangularToPolar RecToPolConverter;

        private int dataWidth;
        private int dataHeight;

        private ObservableCollection<PolarPoint> obstaclePoints;
        private ObservableCollection<PolarPoint> controlPoints;
        private PolarSplineCalculator obstaclePointCalculator;
        private bool editingObstacles;
        private Point obstacleCentre;

        private const int numObstaclePoints = 80;
        private const float boundaryTop = 0.55f;
        private const float boundaryLeft = 0.15f;
        private const float boundaryHeight = 0.1f;
        private const float boundaryWidth = 0.1f;


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

        public bool EditingObstacles
        {
            get => editingObstacles;
            set
            {
                editingObstacles = value;
                OnPropertyChanged(this, nameof(EditingObstacles));
                OnPropertyChanged(this, nameof(EditObstaclesButtonText));
            }
        }
        public ObservableCollection<PolarPoint> ObstaclePoints { get => obstaclePoints; }

        public ObservableCollection<PolarPoint> ControlPoints { get => controlPoints; }
        public Point ObstacleCentre
        {
            get => obstacleCentre;
            set
            {
                obstacleCentre = value;
                OnPropertyChanged(this, nameof(ObstacleCentre));
            }
        }

        public VisualisationControl VisualisationControl { get => visualisationControl; }
        public UnitConversionPanel UnitsPanel { get => unitsPanel; }
        
        public float VisFPS { get => 1 / visFrameTimeAverage.Average; }
        public float BackFPS { get => 1 / backFrameTimeAverage.Average; }
        public float DragCoefficient { get => dragCoefficientAverage.Average; }
        
        public CancellationTokenSource BackendCTS { get => backendCTS; set => backendCTS = value; }

        public string BackendButtonText
        {
            get
            {
                return BackendStatus switch
                {
                    BackendStatus.Running => "Pause simulation",
                    BackendStatus.Stopped => "Resume simulation",
                    _ => string.Empty,
                };
            }
        }

        public string EditObstaclesButtonText
        {
            get
            {
                return EditingObstacles ? "Finish editing" : "Edit simulation obstacles";
            }
        }

        public BackendStatus BackendStatus
        {
            get => backendManager.BackendStatus;
        }

        public Commands.SwitchPanel SwitchPanelCommand { get; set; }
        public Commands.PauseResumeBackend BackendCommand { get; set; }
        public Commands.ChangeWindow ChangeWindowCommand { get; set; }
        public Commands.CreatePopup CreatePopupCommand { get; set; }
        public Commands.EditObstacles EditObstaclesCommand { get; set; }

        public Commands.SimScreenBack BackCommand { get; set; }

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

        public event CancelEventHandler StopBackendExecuting;
        #endregion

        public SimulationScreenVM(ParameterHolder parameterHolder, UnitHolder unitHolder) : base(parameterHolder, unitHolder)
        {
            #region Parameters related to View
            currentButton = null; // Initially no panel selected
            obstaclePoints = new ObservableCollection<PolarPoint>();
            controlPoints = new ObservableCollection<PolarPoint>();
            obstacleCentre = new Point(50, 50);
            obstaclePointCalculator = new PolarSplineCalculator();
            CreateDefaultObstacle();
            controlPoints.CollectionChanged += OnControlPointsChanged;
            editingObstacles = false;
            InVel = parameterHolder.InflowVelocity.Value;
            Chi = parameterHolder.SurfaceFriction.Value;

            unitsPanel = new UnitConversionPanel(unitHolder);
            SwitchPanelCommand = new Commands.SwitchPanel(this);
            BackendCommand = new Commands.PauseResumeBackend(this);
            EditObstaclesCommand = new Commands.EditObstacles(this);
            BackCommand = new Commands.SimScreenBack(this);
            ChangeWindowCommand = new Commands.ChangeWindow();
            CreatePopupCommand = new Commands.CreatePopup();
            RecToPolConverter = new RectangularToPolar();
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
            EmbedObstacles();

            Task.Run(StartComputation);
            backFrameTimeAverage = new MovingAverage<float>(DefaultParameters.FPS_WINDOW_SIZE);
            dragCoefficientAverage = new MovingAverage<float>(DefaultParameters.DRAG_COEF_WINDOW_SIZE);
            backendManager.PropertyChanged += HandleBackendPropertyChanged;
            #endregion

            #region Parameters related to Visualisation
            SetFieldDefaults();
            selectedField = SelectedField.Velocity; // Velocity selected initially.
            SwitchFieldParameters();

            visualisationControl = new VisualisationControl(parameterHolder, streamFunction, dataWidth, dataHeight); // Content of VisualisationControlHolder is bound to this.
            visualisationControl.PropertyChanged += VisFPSUpdate; // FPS updating method
            visFrameTimeAverage = new MovingAverage<float>(DefaultParameters.FPS_WINDOW_SIZE);
            #endregion
        }

        private void OnControlPointsChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            switch (e.Action)
            {
                case NotifyCollectionChangedAction.Add:
                    foreach (object? point in e.NewItems)
                    {
                        if (point is not PolarPoint polarPoint)
                        {
                            throw new ArgumentException("The item added to the collection was not valid.");
                        }
                        obstaclePointCalculator.AddControlPoint(polarPoint);
                    }
                    break;
                    
                case NotifyCollectionChangedAction.Remove:
                    foreach (object? point in e.OldItems)
                    {
                        if (point is not PolarPoint polarPoint)
                        {
                            throw new ArgumentException("The item removed from the collection was not valid");
                        }
                        obstaclePointCalculator.RemoveControlPoint(polarPoint);
                    }
                    break;

                case NotifyCollectionChangedAction.Replace:
                    if (e.OldItems[0] is not PolarPoint oldPoint || e.NewItems[0] is not PolarPoint newPoint)
                    {
                        throw new ArgumentException("The item removed from the collection was not valid");
                    }
                    obstaclePointCalculator.ModifyControlPoint(oldPoint, newPoint); // Check if NewItems contains the new item here.
                    break;
                default:
                    throw new InvalidOperationException("Only add, modify and remove are supported for obstacle points collection.");
            }
            // If control has reached this point, a valid modification has been made to the control points collection
            PlotObstaclePoints();
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

        public void StartComputation()
        {
            try
            {
                backendCTS = new CancellationTokenSource();
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

        private void CreateDefaultObstacle()
        {
            double scale = 10;

            // Define the bean.
            controlPoints.Add(new PolarPoint(scale, Math.PI / 4));
            controlPoints.Add(new PolarPoint(scale, 3 * Math.PI / 4));
            controlPoints.Add(new PolarPoint(scale, 5 * Math.PI / 4));
            controlPoints.Add(new PolarPoint(scale, 7 * Math.PI / 4));
            controlPoints.Add(new PolarPoint(scale / Math.Sqrt(2), Math.PI / 2));

            foreach (PolarPoint controlPoint in controlPoints)
            {
                obstaclePointCalculator.AddControlPoint(controlPoint);
            }

            PlotObstaclePoints();
        }

        /// <summary>
        /// Wipes the <see cref="ObstaclePoints"/> collection and plots new points based on the <see cref="obstaclePointCalculator"/> function.
        /// </summary>
        private void PlotObstaclePoints()
        {
            ObstaclePoints.Clear();
            for (double i = 0; i < numObstaclePoints; i++)
            {
                ObstaclePoints.Add(new PolarPoint(obstaclePointCalculator.CalculatePoint(i / numObstaclePoints * 2 * Math.PI), i / numObstaclePoints * 2 * Math.PI));
            }
            OnPropertyChanged(this, nameof(ObstaclePoints));
        }

        public void EmbedObstacles()
        {
            bool[] obstacles = new bool[(dataWidth + 2) * (dataHeight + 2)];
            for (int i = 1; i <= dataWidth; i++)
            {
                for (int j = 1; j <= dataHeight; j++)
                {
                    obstacles[i * (dataHeight + 2) + j] = true; // Set cells to fluid
                }
            }

            for (int i = 1; i <= dataWidth; i++)
            {
                for (int j = 1; j <= dataHeight; j++)
                {
                    float screenX = i * (float)100 / dataWidth;
                    float screenY = j * (float)100 / dataHeight;
                    PolarPoint polarPoint = (PolarPoint)RecToPolConverter.Convert(new Point(screenX, screenY), typeof(PolarPoint), ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture);
                    if (polarPoint.Radius < obstaclePointCalculator.CalculatePoint(polarPoint.Angle)) // Within the obstacle
                    {
                        obstacles[i * (dataHeight + 2) + j] = false; // Set cells to obstacle
                    }
                }
            }
            _ = backendManager.SendObstacles(obstacles);
        }

        public void CloseBackend()
        {
            if (!backendManager.CloseBackend())
            {
                backendManager.ForceCloseBackend();
            }
        }

        private void VisFPSUpdate(object? sender, PropertyChangedEventArgs e)
        {
            visFrameTimeAverage.UpdateAverage(visualisationControl.FrameTime);
            OnPropertyChanged(this, nameof(VisFPS));
        }

        private void BackFPSUpdate()
        {
            backFrameTimeAverage.UpdateAverage(backendManager.FrameTime);
            OnPropertyChanged(this, nameof(BackFPS));
        }

        private void DragCoefficientUpdate()
        {
            dragCoefficientAverage.UpdateAverage(backendManager.DragCoefficient);
            OnPropertyChanged(this, nameof(DragCoefficient));
        }

        private void HandleBackendPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(backendManager.BackendStatus))
            {
                OnPropertyChanged(sender, nameof(BackendStatus));
                OnPropertyChanged(this, nameof(BackendButtonText));
            } 
            else if (e.PropertyName == nameof(backendManager.FrameTime))
            {
                BackFPSUpdate();
            }
            else if (e.PropertyName == nameof(backendManager.DragCoefficient))
            {
                DragCoefficientUpdate();
            }
        }
    }
}
