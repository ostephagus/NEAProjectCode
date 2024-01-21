using System;
using System.Windows;
using System.Windows.Input;
using UserInterface.ViewModels;
using UserInterface.Views;

namespace UserInterface.HelperClasses
{
    public class Commands
    {
        /// <summary>
        /// Base class for commands that are related to a specific ViewModel using dependency injection. Abstractly implements <see cref="ICommand"/>.
        /// </summary>
        /// <typeparam name="VMType">The type of the ViewModel that will be used with the Command.</typeparam>
        public abstract class VMCommandBase<VMType> : ICommand
        {
            protected VMType parentViewModel;

            public event EventHandler? CanExecuteChanged
            {
                add { }
                remove { }
            }

            public bool CanExecute(object? parameter) { return true; }

            public abstract void Execute(object? parameter);

            public VMCommandBase(VMType parentViewModel)
            {
                this.parentViewModel = parentViewModel;
            }
        }

        /// <summary>
        /// Base class for commands that deal with parameters, again using dependency injection to get the <see cref="ParameterHolder" />.
        /// </summary>
        /// <typeparam name="VMType">The type fo the ViewModel that will be used with the Command.</typeparam>
        public abstract class ParameterCommandBase<VMType> : VMCommandBase<VMType>
        {
            protected ParameterHolder parameterHolder;

            public ParameterCommandBase(VMType parentViewModel, ParameterHolder parameterHolder) : base(parentViewModel)
            {
                this.parameterHolder = parameterHolder;
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

        public class SaveParameters : ParameterCommandBase<AdvancedParametersVM>
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

            public SaveParameters(AdvancedParametersVM parentViewModel, ParameterHolder parameterHolder, ChangeWindow changeWindowCommand) : base(parentViewModel, parameterHolder)
            {
                this.changeWindowCommand = changeWindowCommand;
            }
        }

        public class SwitchPanel : VMCommandBase<SimulationScreenVM>
        {
            public override void Execute(object? parameter)
            {
                string name = ((FrameworkElement)parameter).Name;
                if (name == parentViewModel.CurrentButton) // If the button of the currently open panel is clicked, set the current button to null to close all panels (toggle functionality).
                {
                    parentViewModel.CurrentButton = null;
                }
                else
                {
                    parentViewModel.CurrentButton = name; // If any other panel is open, or no panel is open, open the one corresponding to the button.
                }
            }

            public SwitchPanel(SimulationScreenVM parentViewModel) : base(parentViewModel) { }
        }

        public class ChangeWindow : ICommand
        {
            public event EventHandler? CanExecuteChanged;

            public bool CanExecute(object? parameter) { return true; } // Unless app logic changes, this command can always execute.

            public void Execute(object? parameter)
            {
                if (parameter == null) { return; }
                App.RaiseUserControlChanged(this, new UserControlChangeEventArgs((WindowChangeParameter)parameter));
            }
        }
        public class StopBackend : VMCommandBase<SimulationScreenVM>
        {
            public override void Execute(object? parameter)
            {
                parentViewModel.BackendCTS.Cancel();
            }

            public StopBackend(SimulationScreenVM parentViewModel) : base(parentViewModel) { }
        }
    }

    public struct WindowChangeParameter
    {
        public Type NewWindow { get; set; }
        public bool IsPopup { get; set; }
    }
}
