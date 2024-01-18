using System.Windows;
using System.Windows.Controls;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for ResizableCentredTextBox.xaml
    /// </summary>
    public partial class ResizableCentredTextBox : UserControl
    {
        public string Text
        {
            get { return (string)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Text.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty TextProperty =
            DependencyProperty.Register("Text", typeof(string), typeof(ResizableCentredTextBox), new PropertyMetadata("Text not bound."));

        public ResizableCentredTextBox()
        {
            InitializeComponent();
            LayoutRoot.DataContext = this;
        }
    }
}
