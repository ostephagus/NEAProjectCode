using System;
using System.Windows.Input;
using UserInterface.ViewModels;
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

        public class ResetCommand : ICommand
        {
            private AdvancedParametersVM parameterVM;
            private ParameterHolder parameterHolder;

            public event EventHandler? CanExecuteChanged
            {
                add { }
                remove { }
            }

            public bool CanExecute(object? parameter) { return true; } // Unless app logic changes, this command can always execute.

            public void Execute(object? parameter)
            {
                parameterHolder.TimeStepSafetyFactor.Reset();
                parameterVM.Tau = parameterHolder.TimeStepSafetyFactor.DefaultValue;

                parameterHolder.RelaxationParameter.Reset();
                parameterVM.Omega = parameterHolder.RelaxationParameter.DefaultValue;

                parameterHolder.PressureResidualTolerance.Reset();
                parameterVM.RMax = parameterHolder.PressureResidualTolerance.DefaultValue;

                parameterHolder.PressureMaxIterations.Reset();
                parameterVM.IterMax = parameterHolder.PressureMaxIterations.DefaultValue;
            }

            public ResetCommand(AdvancedParametersVM parameterVM, ParameterHolder parameterHolder)
            {
                this.parameterVM = parameterVM;
                this.parameterHolder = parameterHolder;
            }
        }
    }

    public struct WindowChangeParameter
    {
        public Type NewWindow { get; set; }
        public bool IsPopup { get; set; }
    }
}
