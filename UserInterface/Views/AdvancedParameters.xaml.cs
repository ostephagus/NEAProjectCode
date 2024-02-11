using System.Windows.Controls;
using UserInterface.HelperClasses;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for AdvancedParameters.xaml
    /// </summary>
    public partial class AdvancedParameters : UserControl
    {
        public AdvancedParameters(ParameterHolder parameterHolder, UnitHolder unitHolder)
        {
            InitializeComponent();
            DataContext = new AdvancedParametersVM(parameterHolder, unitHolder);
        }
    }
}
