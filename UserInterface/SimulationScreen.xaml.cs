using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
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

        public event PropertyChangedEventHandler? PropertyChanged;

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

        public enum SidePanelButton //Different side panels on SimluationScreen
        {
            btnParametersSelect,
            btnUnitsSelect,
            btnVisualisationSettingsSelect,
            btnRecordingSelect
        }
    }
}
