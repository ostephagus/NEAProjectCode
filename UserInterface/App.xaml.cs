using System;
using System.Windows;
using System.Windows.Controls;
using UserInterface.HelperClasses;
using UserInterface.ViewModels;
using UserInterface.Views;

#pragma warning disable CS8618 // Compiler doesn't understand that Start() is functionally the constructor for this class.

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private UserControl currentUserControl;
        private UserControl? currentPopup;
        private MainWindow fullScreenWindowContainer; // 2 different container windows to allow for usercontrols to either be popups (that don't take up the whole screen), or fullscreen
        private PopupWindow popupWindowContainer;
        private ParameterHolder parameterHolder;

        public static event EventHandler<UserControlChangeEventArgs>? UserControlChanged;
        public static event EventHandler<UserControlChangeEventArgs>? PopupCreated;
        public static event EventHandler<EventArgs>? PopupDeleted;

        private void ChangeUserControl(object? sender, UserControlChangeEventArgs e)
        {
            currentUserControl = (UserControl)Activator.CreateInstance(e.NewUserControlType, [parameterHolder]); // Use the Type parameter to create a new instance

            fullScreenWindowContainer.Content = currentUserControl;

        }

        private void CreatePopup(object? sender, UserControlChangeEventArgs e)
        {
            currentPopup = (UserControl)Activator.CreateInstance(e.NewUserControlType, [parameterHolder]);
            popupWindowContainer.Content = currentPopup;

            popupWindowContainer.Show();
        }

        private void DeletePopup(object? sender, EventArgs e)
        {
            currentPopup = null;
            popupWindowContainer.Content = currentPopup;

            popupWindowContainer.Hide();
        }

        // Static method for other classes to invoke events without the App instance.
        public static void RaiseUserControlChanged(object? sender, UserControlChangeEventArgs e) 
        {
            UserControlChanged.Invoke(sender, e);
        }

        public static void RaisePopupCreated(object? sender, UserControlChangeEventArgs e)
        {
            PopupCreated.Invoke(sender, e);
        }

        public static void RaisePopupDeleted(object? sender, EventArgs e)
        {
            PopupDeleted.Invoke(sender, e);
        }

        public void Start(object Sender, StartupEventArgs e)
        {
            fullScreenWindowContainer = new MainWindow(); // Initialise container windows
            popupWindowContainer = new PopupWindow
            {
                Height = 400,
                Width = 700
            };

            parameterHolder = new(DefaultParameters.WIDTH, DefaultParameters.HEIGHT, DefaultParameters.TIMESTEP_SAFETY_FACTOR, DefaultParameters.RELAXATION_PARAMETER, DefaultParameters.PRESSURE_RESIDUAL_TOLERANCE, DefaultParameters.PRESSURE_MAX_ITERATIONS, DefaultParameters.REYNOLDS_NUMBER, DefaultParameters.FLUID_VISCOSITY, DefaultParameters.FLUID_VELOCITY, DefaultParameters.SURFACE_FRICTION, new FieldParameters(), DefaultParameters.DRAW_CONTOURS, DefaultParameters.CONTOUR_TOLERANCE, DefaultParameters.CONTOUR_SPACING); // Use the defaults from DefaultParameters constant holder

            currentUserControl = new ConfigScreen(parameterHolder);
            fullScreenWindowContainer.Content = currentUserControl;
            fullScreenWindowContainer.Show();

            UserControlChanged += ChangeUserControl;
            PopupCreated += CreatePopup;
            PopupDeleted += DeletePopup;
        }

        private void Application_Exit(object sender, ExitEventArgs e)
        {
            if (currentUserControl is SimulationScreen simulationScreen) // Close the backend if it is running when application exits (current screen will be SimulationScreen).
            {
                simulationScreen.ViewModel.CloseBackend();
            }
        }

        //public void StartVisualisationDebugging(object sender, StartupEventArgs e)
        //{
        //    fullScreenWindowContainer = new MainWindow();
        //    fullScreenWindowContainer.Content = new VisualisationControl();
        //    fullScreenWindowContainer.Show();
        //}
    }

    public class UserControlChangeEventArgs : EventArgs // EventArgs derivative containing the typename of the new user control
    {
        public Type NewUserControlType { get; }
        public UserControlChangeEventArgs(Type newUserControlType) : base()
        {
            NewUserControlType = newUserControlType;
        }
    }
}
