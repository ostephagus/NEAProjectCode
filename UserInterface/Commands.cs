using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace UserInterface
{
    public class Commands
    {
        public class Command_ChangeWindow : ICommand
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
    public struct WindowChangeParameter
    {
        public Type newWindow;
        public bool isPopup;
    }
}
