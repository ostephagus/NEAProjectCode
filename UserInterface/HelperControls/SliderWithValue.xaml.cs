using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using UserInterface.HelperClasses;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for SliderWithValue.xaml
    /// </summary>
    public partial class SliderWithValue : UserControl
    {

        // Dependency properties are used extensively here to allow for bindings on Value, Minimum and Maximum.

        public double Value
        {
            get { return (double)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty =
            DependencyProperty.Register("Value", typeof(double), typeof(SliderWithValue), new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault)); // Value is two-way: changes to the slider must be passed to the source that uses this UserControl.

        public double Minimum
        {
            get { return (double)GetValue(MinimumProperty); }
            set { SetValue(MinimumProperty, value); }
        }
        public static readonly DependencyProperty MinimumProperty =
            DependencyProperty.Register("Minimum", typeof(double), typeof(SliderWithValue), new PropertyMetadata(0d));

        public double Maximum
        {
            get { return (double)GetValue(MaximumProperty); }
            set { SetValue(MaximumProperty, value); }
        }
        public static readonly DependencyProperty MaximumProperty =
            DependencyProperty.Register("Maximum", typeof(double), typeof(SliderWithValue), new PropertyMetadata(1d));

        public SliderWithValue()
        {
            InitializeComponent();
            LayoutRoot.DataContext = this;
        }

        public bool ForceIntegers { get; set; } = false;

        public TickPlacement TickPlacement { get; } = TickPlacement.None;
    }
}
