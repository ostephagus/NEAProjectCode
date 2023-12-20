using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for SliderWithValue.xaml
    /// </summary>
    public partial class SliderWithValue : UserControl, INotifyPropertyChanged
    {
        public SliderWithValue()
        {
            InitializeComponent();
            DataContext = this;

            slider.ValueChanged += RaisePropertyChanged;
        }
        public int minimum { get; set; } = 0;
        public int maximum { get; set; } = 100;

        private void RaisePropertyChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (PropertyChanged is null) return; // Avoid race condition where property changes as window switches
            PropertyChanged.Invoke(sender, new PropertyChangedEventArgs(nameof(Value)));
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        public double Value { get => slider.Value; set => slider.Value = value; }

        public bool forceIntegers { get; set; } = false;

        public TickPlacement tickPlacement { get; } = TickPlacement.None;
    }
}
