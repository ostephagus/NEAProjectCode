using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

#pragma warning disable CS8618 // Compiler doesn't understand that Start() is functionally the constructor for this class.

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private UserControl currentUserControl;
        private Window currentWindow;
        private MainWindow fullScreenWindowContainer; // 2 different container windows to allow for usercontrols to either be popups (that don't take up the whole screen), or fullscreen
        private PopupWindow popupWindowContainer;

        public static event EventHandler<UserControlChangeEventArgs>? UserControlChanged;

        private void ChangeUserControl(object? sender, UserControlChangeEventArgs e)
        {
            currentUserControl = (UserControl)Activator.CreateInstance(e.NewUserControlType); // Use the Type parameter to create a new instance
            if (e.IsPopup)
            {
                popupWindowContainer.Content = currentUserControl;
                if (currentWindow != popupWindowContainer) // If the currently shown container window is the wrong one, swap them over
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

        public static void RaiseUserControlChanged(object? sender, UserControlChangeEventArgs e) // Static method for other classes to invoke the UserControlChanged event
        {
            UserControlChanged.Invoke(sender, e);
        }

        public void Start(object Sender, StartupEventArgs e)
        {
            fullScreenWindowContainer = new MainWindow(); // Initialise container windows
            popupWindowContainer = new PopupWindow();
            currentUserControl = new ConfigScreen();
            currentWindow = popupWindowContainer;
            popupWindowContainer.Content = currentUserControl;
            popupWindowContainer.Height = 400;
            popupWindowContainer.Width = 700;
            popupWindowContainer.Show();

            UserControlChanged += ChangeUserControl;
        }

        public void StartVisualisationDebugging(object sender, StartupEventArgs e)
        {
            fullScreenWindowContainer = new MainWindow();
            fullScreenWindowContainer.Content = new VisualisationControl();
            fullScreenWindowContainer.Show();
        }
    }

    public class UserControlChangeEventArgs : EventArgs // EventArgs child containing parameters for changing UserControl
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
            NewUserControlType = parameter.NewWindow;
            IsPopup = parameter.IsPopup;
        }
    }
}
