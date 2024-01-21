using System;
using System.Diagnostics;
using System.ComponentModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using UserInterface.HelperClasses;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : SwappableScreen
    {
        public SimulationScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            DataContext = new SimulationScreenVM(parameterHolder);
        }
    }
}
