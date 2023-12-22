using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Input;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for AdvancedParameters.xaml
    /// </summary>
    public partial class AdvancedParameters : SwappableScreen
    {
        public AdvancedParameters(ParameterHolder parameterHolder) : base(parameterHolder) // Sets the parameter holder
        {
            InitializeComponent();
            DataContext = this;
            SetSliders();
        }

        public ICommand Command_ChangeWindow { get; } = new Commands.ChangeWindow();

        private void SetSliders()
        {
            sliderTau.Value = parameterHolder.TimeStepSafetyFactor.Value;
            sliderOmega.Value = parameterHolder.RelaxationParameter.Value;
            sliderR.Value = parameterHolder.PressureResidualTolerance.Value;
            sliderIterMax.Value = parameterHolder.PressureMaxIterations.Value;
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            parameterHolder.TimeStepSafetyFactor.Reset();
            sliderTau.Value = parameterHolder.TimeStepSafetyFactor.DefaultValue;

            parameterHolder.RelaxationParameter.Reset();
            sliderOmega.Value = parameterHolder.RelaxationParameter.DefaultValue;

            parameterHolder.PressureResidualTolerance.Reset();
            sliderR.Value = parameterHolder.PressureResidualTolerance.DefaultValue;

            parameterHolder.PressureMaxIterations.Reset();
            sliderIterMax.Value = parameterHolder.PressureMaxIterations.DefaultValue;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            parameterHolder.TimeStepSafetyFactor.Value = (float)sliderTau.Value;
            parameterHolder.RelaxationParameter.Value = (float)sliderOmega.Value;
            parameterHolder.PressureResidualTolerance.Value = (float)sliderR.Value;
            parameterHolder.PressureMaxIterations.Value = (float)sliderIterMax.Value;

            Command_ChangeWindow.Execute(new WindowChangeParameter() { IsPopup = true, NewWindow = typeof(ConfigScreen) });
        }
    }
}
