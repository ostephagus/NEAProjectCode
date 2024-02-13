using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using UserInterface.Converters;
using UserInterface.HelperClasses;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for SliderWithValue.xaml
    /// </summary>
    public partial class SliderWithValue : UserControl, INotifyPropertyChanged
    {

        // Dependency properties are used extensively here to allow for bindings on Value, Minimum and Maximum.

        public double Value
        {
            get { return (double)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty =
            DependencyProperty.Register("Value", typeof(double), typeof(SliderWithValue), new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault)); // Value is two-way: changes to the slider must be passed to the source that uses this UserControl.
        public double ConvertedValue
        {
            get
            {
                Units UnitConverter = new Units(Unit);
                return (double)UnitConverter.ConvertBack(Value, typeof(double), 0, System.Globalization.CultureInfo.CurrentCulture);
            }
            set
            {
                Units UnitConverter = new Units(Unit);
                Value = (double)UnitConverter.Convert(value, typeof(double), 0, System.Globalization.CultureInfo.CurrentCulture);

            }
        }

        public double Minimum
        {
            get { return (double)GetValue(MinimumProperty); }
            set { SetValue(MinimumProperty, value); }
        }
        public static readonly DependencyProperty MinimumProperty =
            DependencyProperty.Register("Minimum", typeof(double), typeof(SliderWithValue), new PropertyMetadata(0d));
        public double ConvertedMinimum
        {
            get
            {
                Units UnitConverter = new Units(Unit);
                return (double)UnitConverter.ConvertBack(Minimum, typeof(double), 0, System.Globalization.CultureInfo.CurrentCulture);
            }
            set
            {
                Units UnitConverter = new Units(Unit);
                Minimum = (double)UnitConverter.Convert(value, typeof(double), 0, System.Globalization.CultureInfo.CurrentCulture);

            }
        }

        public double Maximum
        {
            get { return (double)GetValue(MaximumProperty); }
            set { SetValue(MaximumProperty, value); }
        }
        public static readonly DependencyProperty MaximumProperty =
            DependencyProperty.Register("Maximum", typeof(double), typeof(SliderWithValue), new PropertyMetadata(1d));
        public double ConvertedMaximum
        {
            get
            {
                Units UnitConverter = new Units(Unit);
                return (double)UnitConverter.ConvertBack(Maximum, typeof(double), 0, System.Globalization.CultureInfo.CurrentCulture);
            }
            set
            {
                Units UnitConverter = new Units(Unit);
                Maximum = (double)UnitConverter.Convert(value, typeof(double), 0, System.Globalization.CultureInfo.CurrentCulture);
            }
        }

        public string UnitShortName { get => Unit?.ShortName ?? ""; }


        public UnitClasses.Unit Unit
        {
            get { return (UnitClasses.Unit)GetValue(UnitProperty); }
            set
            {
                SetValue(UnitProperty, value);
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ConvertedValue)));
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ConvertedMinimum)));
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ConvertedMaximum)));
            }
        }

        public static readonly DependencyProperty UnitProperty =
            DependencyProperty.Register(nameof(Unit), typeof(UnitClasses.Unit), typeof(SliderWithValue));

        public event PropertyChangedEventHandler? PropertyChanged;

        public SliderWithValue()
        {
            InitializeComponent();
            LayoutRoot.DataContext = this;
        }

        public bool ForceIntegers { get; set; } = false;

        public TickPlacement TickPlacement { get; } = TickPlacement.None;
    }
}
