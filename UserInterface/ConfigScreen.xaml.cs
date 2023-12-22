using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows.Controls;
using System.Windows.Input;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for ConfigScreen.xaml
    /// </summary>
    public partial class ConfigScreen : SwappableScreen
    {
        private void SetSliders()
        {
            SliderInVel.Value = parameterHolder.FluidVelocity.Value;
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
            parameterHolder.FluidVelocity.Value = (float)SliderInVel.Value;
            Trace.WriteLine(parameterHolder.FluidVelocity.Value);
        }

        private void SliderChi_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.SurfaceFriction.Value = (float)SliderChi.Value;
        }

        private void SliderWidth_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.Width.Value = (float)SliderWidth.Value;
        }

        private void SliderHeight_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.Height.Value = (float)SliderHeight.Value;
        }

        private void BtnReset_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            parameterHolder.FluidVelocity.Reset();
            SliderInVel.Value = parameterHolder.FluidVelocity.DefaultValue;

            parameterHolder.SurfaceFriction.Reset();
            SliderChi.Value = parameterHolder.SurfaceFriction.DefaultValue;

            parameterHolder.Width.Reset();
            SliderWidth.Value = parameterHolder.Width.DefaultValue;

            parameterHolder.Height.Reset();
            SliderHeight.Value = parameterHolder.Height.DefaultValue;
        }
    }
}
