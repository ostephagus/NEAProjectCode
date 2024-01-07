using System;
using System.Windows.Input;
using UserInterface.Views;

namespace UserInterface.HelperClasses
{
    public class Commands
    {
        public class ChangeWindow : ICommand
        {
            public event EventHandler? CanExecuteChanged
            {
                add { }
                remove { }
            }

            public bool CanExecute(object? parameter) { return true; } // Unless app logic changes, this command can always execute.

            public void Execute(object? parameter)
            {
                if (parameter == null) { return; }
                App.RaiseUserControlChanged(this, new UserControlChangeEventArgs((WindowChangeParameter)parameter));
            }
        }
        public class StopBackend : ICommand
        {
            public event EventHandler? CanExecuteChanged
            {
                add { }
                remove { }
            }

            public bool CanExecute(object? parameter) { return true; } // Unless app logic changes, this command can always execute.

            public void Execute(object? parameter)
            {
                SimulationScreen.RaiseStopBackendExecuting();
            }
        }
    }
    public struct WindowChangeParameter
    {
        public Type NewWindow { get; set; }
        public bool IsPopup { get; set; }
    }
}
