using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for SliderWithValue.xaml
    /// </summary>
    public partial class SliderWithValue : UserControl, INotifyPropertyChanged
    {
        public SliderWithValue()
        {
            InitializeComponent();
            LayoutRoot.DataContext = this;

            // Add event handler to update the Value property when the Slider value changes
            slider.ValueChanged += Slider_ValueChanged;
        }
        public double Minimum { get; set; } = 0;
        public double Maximum { get; set; } = 100;

        public event PropertyChangedEventHandler? PropertyChanged;

        public static readonly DependencyProperty ValueProperty =
            DependencyProperty.Register(
                "Value",
                typeof(double),
                typeof(SliderWithValue),
                new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault, OnValueChanged));

        public double Value
        {
            get { return (double)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }

        private static void OnValueChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            SliderWithValue sliderWithValue = (SliderWithValue)d; // Get the calling instance
            sliderWithValue.slider.Value = (double)e.NewValue;
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // Update the Value property when the Slider value changes
            Value = e.NewValue;
        }

        public bool ForceIntegers { get; set; } = false;

        public TickPlacement TickPlacement { get; } = TickPlacement.None;
    }
}
