using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Markup;

namespace UserInterface
{
    public class Commands
    {
        public class ChangeWindow : ICommand
        {
            public event EventHandler? CanExecuteChanged;

            public bool CanExecute(object? parameter) { return true; } //Unless app logic changes, this command can always execute.

            public void Execute(object? parameter)
            {
                if (parameter == null) { return; }
                App.RaiseUserControlChanged(this, new UserControlChangeEventArgs((WindowChangeParameter)parameter));
            }
        }
    }

    public class ShowSidePanel : ICommand
    {
        public event EventHandler? CanExecuteChanged;

        public bool CanExecute(object? parameter) { return true; }

        public void Execute(object? parameter)
        {
            //Somehow figure out a way to change the shown screen without access to the instance of the SimulationScreen
        }
    }
    public struct WindowChangeParameter
    {
        public Type NewWindow { get; set; }
        public bool IsPopup { get; set; }
    }

    public enum SidePanel //Different side panels on SimluationScreen
    {
        Parameters,
        AdvancedParameters,
        VisualisationSettings
    }
}
