using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private UserControl currentUserControl;
        private MainWindow containerWindow;

        public static event EventHandler<UserControlChangeEventArgs>? UserControlChanged;

        private void ChangeUserControl(object sender, UserControlChangeEventArgs e)
        {
            currentUserControl = (UserControl)Activator.CreateInstance(e.NewUserControlType);
            containerWindow.Content = currentUserControl;
        }

        public static void RaiseUserControlChanged(object sender, UserControlChangeEventArgs e)
        {
            UserControlChanged.Invoke(sender, e);
        }
    }

    public class UserControlChangeEventArgs : EventArgs
    {
        public Type NewUserControlType { get; }
        public UserControlChangeEventArgs(Type newUserControlType)
        {
            NewUserControlType = newUserControlType;
        }
    }
}
