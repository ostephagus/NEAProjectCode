using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;

namespace UserInterface.Views
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
        public double minimum { get; set; } = 0;
        public double maximum { get; set; } = 100;

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
