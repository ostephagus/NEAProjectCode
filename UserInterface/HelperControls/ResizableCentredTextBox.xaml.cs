using System.Windows.Controls;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for ResizableCentredTextBox.xaml
    /// </summary>
    public partial class ResizableCentredTextBox : UserControl
    {
        public string Text { get; set; } = "";
        public ResizableCentredTextBox()
        {
            InitializeComponent();
            DataContext = this;
        }
    }
}
