using System;
using System.ComponentModel;
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
            sliderInVel.Value = parameterHolder.FluidVelocity.Value;
            sliderChi.Value = parameterHolder.SurfaceFriction.Value;
            sliderWidth.Value = parameterHolder.Width.Value;
            sliderHeight.Value = parameterHolder.Height.Value;
        }

        public ConfigScreen() : base()
        {
            InitializeComponent();
            DataContext = this;
            SetSliders();
        }

        public ConfigScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            DataContext = this;
            SetSliders();
        }

        public ICommand Command_ChangeWindow { get; } = new Commands.ChangeWindow();

        private void sliderInVel_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.FluidVelocity.Value = (float)sliderInVel.Value;
        }

        private void sliderChi_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.FluidVelocity.Value = (float)sliderChi.Value;
        }

        private void sliderWidth_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.FluidVelocity.Value = (float)sliderWidth.Value;
        }

        private void sliderHeight_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            parameterHolder.FluidVelocity.Value = (float)sliderHeight.Value;
        }

        private void btnReset_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            parameterHolder.FluidVelocity.Reset();
            sliderInVel.Value = parameterHolder.FluidVelocity.DefaultValue;

            parameterHolder.SurfaceFriction.Reset();
            sliderChi.Value = parameterHolder.SurfaceFriction.DefaultValue;

            parameterHolder.Width.Reset();
            sliderWidth.Value = parameterHolder.Width.DefaultValue;

            parameterHolder.Height.Reset();
            sliderHeight.Value = parameterHolder.Height.DefaultValue;
        }
    }
}
