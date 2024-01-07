using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows.Input;
using UserInterface.HelperClasses;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for ConfigScreen.xaml
    /// </summary>
    public partial class ConfigScreen : SwappableScreen
    {
        private void SetSliders()
        {
            SliderInVel.Value = parameterHolder.InflowVelocity.Value;
            SliderChi.Value = parameterHolder.SurfaceFriction.Value;
            SliderWidth.Value = parameterHolder.Width.Value;
            SliderHeight.Value = parameterHolder.Height.Value;
        }

        public ConfigScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            DataContext = this;
            SetSliders();
        }

        public ICommand Command_ChangeWindow { get; } = new Commands.ChangeWindow();

        private void SliderInVel_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, (float)SliderInVel.Value);
            Trace.WriteLine(parameterHolder.InflowVelocity.Value);
        }

        private void SliderChi_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, (float)SliderChi.Value);
        }

        private void SliderWidth_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.Width = ModifyParameterValue(parameterHolder.Width, (float)SliderWidth.Value);
        }

        private void SliderHeight_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.Height = ModifyParameterValue(parameterHolder.Height, (float)SliderHeight.Value);
        }

        private void BtnReset_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            parameterHolder.InflowVelocity.Reset();
            SliderInVel.Value = parameterHolder.InflowVelocity.DefaultValue;

            parameterHolder.SurfaceFriction.Reset();
            SliderChi.Value = parameterHolder.SurfaceFriction.DefaultValue;

            parameterHolder.Width.Reset();
            SliderWidth.Value = parameterHolder.Width.DefaultValue;

            parameterHolder.Height.Reset();
            SliderHeight.Value = parameterHolder.Height.DefaultValue;
        }
    }
}
