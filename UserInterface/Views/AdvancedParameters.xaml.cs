using System.Diagnostics;
using System.Windows;
using System.Windows.Input;
using UserInterface.HelperClasses;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for AdvancedParameters.xaml
    /// </summary>
    public partial class AdvancedParameters : SwappableScreen
    {
        public AdvancedParameters(ParameterHolder parameterHolder) : base(parameterHolder) // Sets the parameter holder
        {
            InitializeComponent();
            DataContext = new AdvancedParametersVM(parameterHolder);
        }
    }
}
