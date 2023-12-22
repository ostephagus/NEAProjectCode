using System;
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
        private SwappableScreen currentUserControl;
        private Window currentWindow;
        private MainWindow fullScreenWindowContainer; // 2 different container windows to allow for usercontrols to either be popups (that don't take up the whole screen), or fullscreen
        private PopupWindow popupWindowContainer;
        private ParameterHolder parameterHolder;

        public static event EventHandler<UserControlChangeEventArgs>? UserControlChanged;

        private void ChangeUserControl(object? sender, UserControlChangeEventArgs e)
        {
            currentUserControl = (SwappableScreen)Activator.CreateInstance(e.NewUserControlType, new object?[] {parameterHolder}); // Use the Type parameter to create a new instance
            //currentUserControl.ParameterHolder = parameterHolder; // Give the new instance access to the semi-global parameters
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

        private void SetDefaultParameters()
        {
            const float width = 1f;
            const float height = 1f;
            const float timeStepSafetyFactor = 0.8f;
            const float relaxationParameter = 1.7f;
            const float pressureResidualTolerance = 2f;
            const float pressureMaxIterations = 1000f;
            const float reynoldsNumber = 2000f;
            const float fluidVelocity = 1f;
            const float surfaceFriction = 0f;
            FieldParameters initialFieldParams = new();
            const bool drawContours = true;
            const float contourTolerance = 0.01f;
            const float contourSpacing = 0.05f;

            parameterHolder = new(width, height, timeStepSafetyFactor, relaxationParameter, pressureResidualTolerance, pressureMaxIterations, reynoldsNumber, fluidVelocity, surfaceFriction, initialFieldParams, drawContours, contourTolerance, contourSpacing);
        }

        public static void RaiseUserControlChanged(object? sender, UserControlChangeEventArgs e) // Static method for other classes to invoke the UserControlChanged event
        {
            UserControlChanged.Invoke(sender, e);
        }

        public void Start(object Sender, StartupEventArgs e)
        {
            fullScreenWindowContainer = new MainWindow(); // Initialise container windows
            popupWindowContainer = new PopupWindow();

            SetDefaultParameters();

            currentUserControl = new ConfigScreen(parameterHolder);
            currentWindow = popupWindowContainer;
            popupWindowContainer.Content = currentUserControl;
            popupWindowContainer.Height = 400;
            popupWindowContainer.Width = 700;
            popupWindowContainer.Show();

            UserControlChanged += ChangeUserControl;
        }

        //public void StartVisualisationDebugging(object sender, StartupEventArgs e)
        //{
        //    fullScreenWindowContainer = new MainWindow();
        //    fullScreenWindowContainer.Content = new VisualisationControl();
        //    fullScreenWindowContainer.Show();
        //}
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
