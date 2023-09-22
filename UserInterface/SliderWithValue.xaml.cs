using System;
using System.Collections.Generic;
using System.Linq;
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
    public partial class SliderWithValue : UserControl
    {
        public SliderWithValue()
        {
            InitializeComponent();
            DataContext = this;
        }
        public int minimum { get; set; } = 0;
        public int maximum { get; set; } = 100;

        public bool forceIntegers { get; set; } = false;

        public TickPlacement tickPlacement { get; } = TickPlacement.None;

        public ValueConverters.BoolToTickStatus Converter_BoolToTickStatus { get; } = new();
    }
}
