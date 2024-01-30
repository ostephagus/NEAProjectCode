﻿using OpenTK.Graphics.OpenGL;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using UserInterface.Converters;
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
        private MovingAverage<float> visFrameTimeAverage;
        private MovingAverage<float> backFrameTimeAverage;

        private int dataWidth;
        private int dataHeight;

        private List<PolarPoint> obstaclePoints;
        private PolarSplineCalculator obstaclePointCalculator;
        private bool editingObstacles;

        const int numObstaclePoints = 40;
        const float boundaryTop = 0.55f;
        const float boundaryLeft = 0.15f;
        const float boundaryHeight = 0.1f;
        const float boundaryWidth = 0.1f;


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

        public double ObstacleLeft { get; set; } = 0.15;

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
            }
        }
        public List<PolarPoint> ObstaclePoints { get => obstaclePoints; set => obstaclePoints = value; }

        public VisualisationControl VisualisationControl { get => visualisationControl; }
        
        public float VisFPS { get => 1 / visFrameTimeAverage.Average; }
        public float BackFPS { get => 1 / backFrameTimeAverage.Average; }
        
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

        public BackendStatus BackendStatus
        {
            get => backendManager.BackendStatus;
        }

        public Commands.SwitchPanel SwitchPanelCommand { get; set; }
        public Commands.PauseResumeBackend BackendCommand { get; set; }
        public Commands.ChangeWindow ChangeWindowCommand { get; set; }
        public Commands.CreatePopup CreatePopupCommand { get; set; }
        public Commands.EditObstacles EditObstaclesCommand { get; set; }

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

        public SimulationScreenVM(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            #region Parameters related to View
            currentButton = null; // Initially no panel selected
            obstaclePoints = new List<PolarPoint>();
            obstaclePointCalculator = new PolarSplineCalculator();
            editingObstacles = false;
            InVel = parameterHolder.InflowVelocity.Value;
            Chi = parameterHolder.SurfaceFriction.Value;

            CreateDefaultObstacle();

            SwitchPanelCommand = new Commands.SwitchPanel(this);
            BackendCommand = new Commands.PauseResumeBackend(this);
            EditObstaclesCommand = new Commands.EditObstacles(this);
            ChangeWindowCommand = new Commands.ChangeWindow();
            CreatePopupCommand = new Commands.CreatePopup();
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
            backFrameTimeAverage = new MovingAverage<float>(DefaultParameters.FPS_WINDOW_SIZE);
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

            int leftCell = (int)(boundaryLeft * dataWidth);
            int rightCell = (int)((boundaryLeft + boundaryWidth) * dataWidth);
            int bottomCell = (int)((boundaryTop - boundaryHeight) * dataHeight);
            int topCell = (int)(boundaryTop * dataHeight);

            for (int i = leftCell; i < rightCell; i++)
            { // Create a square of boundary cells
                for (int j = bottomCell; j < topCell; j++)
                {
                    obstacles[(i + 1) * (dataHeight + 2) + j + 1] = false;
                }
            }


            return backendManager.SendObstacles(obstacles);
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
            double scale = 0.15;
            obstaclePointCalculator.AddControlPoint(new PolarPoint(scale, Math.PI / 4));
            obstaclePointCalculator.AddControlPoint(new PolarPoint(scale, 3 * Math.PI / 4));
            obstaclePointCalculator.AddControlPoint(new PolarPoint(scale, 5 * Math.PI / 4));
            obstaclePointCalculator.AddControlPoint(new PolarPoint(scale, 7 * Math.PI / 4));
            obstaclePointCalculator.AddControlPoint(new PolarPoint(scale / Math.Sqrt(2), Math.PI / 2));
            for (double i = 0; i < numObstaclePoints; i++)
            {
                ObstaclePoints.Add(new PolarPoint(obstaclePointCalculator.CalculatePoint(i / 40 * 2 * Math.PI), i / 40 * 2 * Math.PI));
            }
        }

        public void EmbedObstacles()
        {
            // Here need to take the positions of the objects on the canvas and use them to populate the Obstacles array, then send it to backend.
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
        }

        private void BackFPSUpdate()
        {
            backFrameTimeAverage.UpdateAverage(backendManager.FrameTime);
            OnPropertyChanged(this, nameof(BackFPS));
        }
    }
}
