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
        }
        public double Minimum { get; set; } = 0;
        public double Maximum { get; set; } = 100;

        public event PropertyChangedEventHandler? PropertyChanged;

        public static readonly DependencyProperty ValueProperty = DependencyProperty.Register("Value", typeof(double), typeof(SliderWithValue));

        public double Value {
            get => (double)GetValue(ValueProperty);
            set
            {
                SetValue(ValueProperty, value);
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Value)));
            }
        }

        public bool ForceIntegers { get; set; } = false;

        public TickPlacement TickPlacement { get; } = TickPlacement.None;
    }
}
