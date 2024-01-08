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

        public ICommand Command_ChangeWindow { get; } = new Commands.ChangeWindow();

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            parameterHolder.TimeStepSafetyFactor.Reset();
            sliderTau.Value = parameterHolder.TimeStepSafetyFactor.DefaultValue;

            parameterHolder.RelaxationParameter.Reset();
            sliderOmega.Value = parameterHolder.RelaxationParameter.DefaultValue;

            parameterHolder.PressureResidualTolerance.Reset();
            sliderRMax.Value = parameterHolder.PressureResidualTolerance.DefaultValue;

            parameterHolder.PressureMaxIterations.Reset();
            sliderIterMax.Value = parameterHolder.PressureMaxIterations.DefaultValue;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            parameterHolder.TimeStepSafetyFactor = ModifyParameterValue(parameterHolder.TimeStepSafetyFactor, (float)sliderTau.Value);
            parameterHolder.RelaxationParameter = ModifyParameterValue(parameterHolder.RelaxationParameter, (float)sliderOmega.Value);
            parameterHolder.PressureResidualTolerance = ModifyParameterValue(parameterHolder.PressureResidualTolerance, (float)sliderRMax.Value);
            parameterHolder.PressureMaxIterations = ModifyParameterValue(parameterHolder.PressureMaxIterations, (float)sliderIterMax.Value);

            Command_ChangeWindow.Execute(new WindowChangeParameter() { IsPopup = true, NewWindow = typeof(ConfigScreen) });
        }
    }
}
