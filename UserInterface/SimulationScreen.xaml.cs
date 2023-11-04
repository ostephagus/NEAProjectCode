using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : UserControl, INotifyPropertyChanged
    {
        private SidePanelButton? currentButton;
        private CancellationTokenSource backendCancellationTokenSource;

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
            Task t = Task.Run(StartComputation); // Asynchronously run the computation
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

        private void StartComputation()
        {
            try
            {
                BackendManager backendManager = new BackendManager();
                backendManager.ConnectBackend();
                // Start the visualisation also
                double[] pressure = new double[backendManager.FieldLength];

                backendManager.GetFieldStreamsAsync(null, null, pressure, null, backendCancellationTokenSource.Token);

            } catch (IOException e)
            {
                MessageBox.Show(e.Message);
            } catch
            {
                MessageBox.Show("Generic error");
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
