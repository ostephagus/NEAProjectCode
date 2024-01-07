using System.Windows.Controls;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for ResizableCentredTextBox.xaml
    /// </summary>
    public partial class ResizableCentredTextBox : UserControl
    {
        public ResizableCentredTextBox()
        {
            InitializeComponent();
            DataContext = this;
        }

        public string Text { get; set; } = "";
    }
}
