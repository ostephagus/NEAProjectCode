using System.Windows.Controls;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for ResizableCentredTextBox.xaml
    /// </summary>
    public partial class ResizableCentredTextBox : UserControl
    {
        public ResizableCentredTextBox()
        {
            InitializeComponent();
        }
        public string Text { get; set; } = "";
    }
}
