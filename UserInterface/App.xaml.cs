using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

#pragma warning disable CS8618 //Compiler doesn't understand that Start() is functionally the constructor for this class.

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private UserControl currentUserControl;
        private Window currentWindow;
        private MainWindow fullScreenWindowContainer;
        private PopupWindow popupWindowContainer;

        public static event EventHandler<UserControlChangeEventArgs>? UserControlChanged;

        private void ChangeUserControl(object? sender, UserControlChangeEventArgs e)
        {
            currentUserControl = (UserControl)Activator.CreateInstance(e.NewUserControlType);
            if (e.IsPopup)
            {
                popupWindowContainer.Content = currentUserControl;
                if (currentWindow != popupWindowContainer)
                {
                    currentWindow.Close();
                    popupWindowContainer.Show();
                    currentWindow = popupWindowContainer;
                }
            }
            else
            {
                fullScreenWindowContainer.Content = currentUserControl;
                if (currentWindow != fullScreenWindowContainer)
                {
                    currentWindow.Close();
                    fullScreenWindowContainer.Show();
                    currentWindow = fullScreenWindowContainer;
                }
            }
        }

        public static void RaiseUserControlChanged(object? sender, UserControlChangeEventArgs e)
        {
            UserControlChanged.Invoke(sender, e);
        }

        public void Start(object Sender, StartupEventArgs e)
        {
            fullScreenWindowContainer = new MainWindow(); //Initialise container windows
            popupWindowContainer = new PopupWindow();
            currentUserControl = new ConfigScreen();
            popupWindowContainer.Content = currentUserControl;
            popupWindowContainer.Show();

            UserControlChanged += ChangeUserControl;
        }
    }

    public class UserControlChangeEventArgs : EventArgs
    {
        public Type NewUserControlType { get; }
        public bool IsPopup { get; }
        public UserControlChangeEventArgs(Type newUserControlType, bool isPopup)
        {
            NewUserControlType = newUserControlType;
            IsPopup = isPopup;
        }

        public UserControlChangeEventArgs(WindowChangeParameter parameter)
        {
            NewUserControlType = parameter.newWindow;
            IsPopup = parameter.isPopup;
        }
    }
}
