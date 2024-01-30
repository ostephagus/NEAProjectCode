using UserInterface.HelperClasses;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : SwappableScreen
    {
        private readonly SimulationScreenVM viewModel;

        public SimulationScreenVM ViewModel => viewModel;

        public SimulationScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            viewModel = new SimulationScreenVM(parameterHolder);
            DataContext = viewModel;
        }

    }
}
