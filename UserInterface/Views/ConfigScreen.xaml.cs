using System.Windows.Controls;
using UserInterface.HelperClasses;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for ConfigScreen.xaml
    /// </summary>
    public partial class ConfigScreen : UserControl
    {
        public ConfigScreen(ParameterHolder parameterHolder, UnitHolder unitHolder)
        {
            InitializeComponent();
            DataContext = new ConfigScreenVM(parameterHolder, unitHolder);
        }
    }
}
