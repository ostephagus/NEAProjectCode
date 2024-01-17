using UserInterface.HelperClasses;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for ConfigScreen.xaml
    /// </summary>
    public partial class ConfigScreen : SwappableScreen
    {
        public ConfigScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            DataContext = new ConfigScreenVM(parameterHolder);
        }
    }
}
