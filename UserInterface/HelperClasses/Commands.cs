using System;
using System.Security.Policy;
using System.Windows.Input;
using UserInterface.ViewModels;
using UserInterface.Views;

namespace UserInterface.HelperClasses
{
    public class Commands
    {
        public abstract class ParameterCommandBase<VMType> : ICommand
        {
            protected VMType parentViewModel;
            protected ParameterHolder parameterHolder;

            public event EventHandler? CanExecuteChanged
            {
                add { }
                remove { }
            }

            public bool CanExecute(object? parameter) { return true; }

            public abstract void Execute(object? parameter);

            public ParameterCommandBase(VMType parentViewModel, ParameterHolder parameterHolder)
            {
                this.parentViewModel = parentViewModel;
                this.parameterHolder = parameterHolder;
            }
        }

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

        public class AdvancedParametersReset : ParameterCommandBase<AdvancedParametersVM>
        {
            public override void Execute(object? parameter)
            {
                parameterHolder.TimeStepSafetyFactor.Reset();
                parentViewModel.Tau = parameterHolder.TimeStepSafetyFactor.DefaultValue;

                parameterHolder.RelaxationParameter.Reset();
                parentViewModel.Omega = parameterHolder.RelaxationParameter.DefaultValue;

                parameterHolder.PressureResidualTolerance.Reset();
                parentViewModel.RMax = parameterHolder.PressureResidualTolerance.DefaultValue;

                parameterHolder.PressureMaxIterations.Reset();
                parentViewModel.IterMax = parameterHolder.PressureMaxIterations.DefaultValue;
            }

            public AdvancedParametersReset(AdvancedParametersVM parentViewModel, ParameterHolder parameterHolder) : base(parentViewModel, parameterHolder) { }
        }

        public class ConfigScreenReset : ParameterCommandBase<ConfigScreenVM>
        {
            public override void Execute(object? parameter)
            {
                parameterHolder.InflowVelocity.Reset();
                parentViewModel.InVel = parameterHolder.InflowVelocity.DefaultValue;

                parameterHolder.SurfaceFriction.Reset();
                parentViewModel.Chi = parameterHolder.SurfaceFriction.DefaultValue;

                parameterHolder.Width.Reset();
                parentViewModel.Width = parameterHolder.Width.DefaultValue;

                parameterHolder.Height.Reset();
                parentViewModel.Height = parameterHolder.Height.DefaultValue;
            }
            
            public ConfigScreenReset(ConfigScreenVM parentViewModel, ParameterHolder parameterHolder) : base(parentViewModel, parameterHolder) { }
        }

        public class SaveCommand : ParameterCommandBase<AdvancedParametersVM>
        {
            private readonly ChangeWindow changeWindowCommand;

            private ParameterStruct<T> ModifyParameterValue<T>(ParameterStruct<T> parameterStruct, T newValue)
            {
                parameterStruct.Value = newValue;
                return parameterStruct;
            }

            public override void Execute(object? parameter)
            {
                parameterHolder.TimeStepSafetyFactor = ModifyParameterValue(parameterHolder.TimeStepSafetyFactor, parentViewModel.Tau);
                parameterHolder.RelaxationParameter = ModifyParameterValue(parameterHolder.RelaxationParameter, parentViewModel.Omega);
                parameterHolder.PressureResidualTolerance = ModifyParameterValue(parameterHolder.PressureResidualTolerance, parentViewModel.RMax);
                parameterHolder.PressureMaxIterations = ModifyParameterValue(parameterHolder.PressureMaxIterations, parentViewModel.IterMax);

                changeWindowCommand.Execute(new WindowChangeParameter() { IsPopup = true, NewWindow = typeof(ConfigScreen) });
            }

            public SaveCommand(AdvancedParametersVM parentViewModel, ParameterHolder parameterHolder, ChangeWindow changeWindowCommand) : base(parentViewModel, parameterHolder)
            {
                this.changeWindowCommand = changeWindowCommand;
            }
        }
    }

    public struct WindowChangeParameter
    {
        public Type NewWindow { get; set; }
        public bool IsPopup { get; set; }
    }
}
