using System;
using System.ComponentModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : UserControl, INotifyPropertyChanged
    {
        private SidePanelButton? currentButton;
        private CancellationTokenSource backendCancellationTokenSource;

        private BackendManager backendManager;

        private float[] horizontalVelocity;
        private float[] pressure;
        private float[] streamFunction;
        private int dataWidth;
        private int dataHeight;

        private int min = 0;
        private int max = 1;
        private float contourTolerance = 0.01f;
        private float contourSpacingMultiplier = 0.1f;

        public event PropertyChangedEventHandler? PropertyChanged;
        public static event CancelEventHandler? StopBackendExecuting;

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

        public SimulationScreen()
        {
            InitializeComponent();
            DataContext = this;
            currentButton = null;
            backendCancellationTokenSource = new CancellationTokenSource();
            StopBackendExecuting += (object? sender, CancelEventArgs e) => backendCancellationTokenSource.Cancel();
            StartComponents();
            Task.Run(StartComputation); // Asynchronously run the computation

        }

        private void panelButton_Click(object sender, RoutedEventArgs e)
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

        private void StartComponents()
        {
            backendManager = new BackendManager();
            backendManager.ConnectBackend();

            horizontalVelocity = new float[backendManager.FieldLength];
            pressure = new float[backendManager.FieldLength];
            streamFunction = new float[backendManager.FieldLength];
            dataWidth = backendManager.IMax;
            dataHeight = backendManager.JMax;

            VisualisationControlHolder.Content = new VisualisationControl(horizontalVelocity, streamFunction, dataWidth, dataHeight, min, max, contourTolerance, contourSpacingMultiplier);
        }

        private void StartComputation()
        {
            try
            {
                backendManager.GetFieldStreamsAsync(horizontalVelocity, null, null, streamFunction, backendCancellationTokenSource.Token);
            } catch (IOException e)
            {
                MessageBox.Show(e.Message);
            } catch (Exception e)
            {
                MessageBox.Show($"Generic error: {e.Message}");
            }
        }

        public static void RaiseStopBackendExecuting()
        {
            StopBackendExecuting.Invoke(null, new CancelEventArgs());
        }

        public enum SidePanelButton //Different side panels on SimluationScreen
        {
            btnParametersSelect,
            btnUnitsSelect,
            btnVisualisationSettingsSelect,
            btnRecordingSelect
        }
    }
}
