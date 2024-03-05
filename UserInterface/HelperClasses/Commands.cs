using Microsoft.Win32;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
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

            public event EventHandler? CanExecuteChanged;

            public virtual void OnCanExecuteChanged(object? sender, EventArgs e)
            {
                CanExecuteChanged?.Invoke(sender, e);
            }

            public virtual bool CanExecute(object? parameter) { return true; }

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

        public class SetAirParameters : VMCommandBase<ConfigScreenVM>
        {
            public override void Execute(object? parameter)
            {
                parentViewModel.ReynoldsNo = DefaultParameters.REYNOLDS_NUMBER;
                parentViewModel.Viscosity = DefaultParameters.FLUID_VISCOSITY;
                parentViewModel.Density = DefaultParameters.FLUID_DENSITY;
            }

            public SetAirParameters(ConfigScreenVM parentViewModel) : base(parentViewModel) { }
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

                App.RaisePopupDeleted(this, new EventArgs());
            }

            public SaveParameters(AdvancedParametersVM parentViewModel, ParameterHolder parameterHolder, ChangeWindow changeWindowCommand) : base(parentViewModel, parameterHolder) { }
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
                App.RaiseUserControlChanged(this, new UserControlChangeEventArgs((Type)parameter));
            }
        }

        public class CreatePopup : ICommand
        {
            public event EventHandler? CanExecuteChanged;

            public bool CanExecute(object? parameter) { return true; }

            public void Execute(object? parameter)
            {
                if (parameter == null) return;
                App.RaisePopupCreated(this, new UserControlChangeEventArgs((Type)parameter));
            }
        }

        public class PauseResumeBackend : VMCommandBase<SimulationScreenVM>
        {
            public override bool CanExecute(object? parameter)
            {
                return !parentViewModel.EditingObstacles; // Cannot execute when editing obstacles.
            }

            public override void Execute(object? parameter)
            {
                switch (parentViewModel.BackendStatus)
                {
                    case BackendStatus.Running:
                        parentViewModel.BackendCTS.Cancel(); // Pause the backend.
                        break;
                    case BackendStatus.Stopped:
                        Task.Run(parentViewModel.StartComputation); // Resume the backend computation.
                        break;
                    default:
                        break;
                }
            }
            public PauseResumeBackend(SimulationScreenVM parentViewModel) : base(parentViewModel)
            {
                parentViewModel.PropertyChanged += VMPropertyChanged;
            }

            private void VMPropertyChanged(object? sender, PropertyChangedEventArgs e)
            {
                if (e.PropertyName == nameof(parentViewModel.EditingObstacles))
                {
                    OnCanExecuteChanged(sender, e);
                }
            }
        }

        public class EditObstacles : VMCommandBase<SimulationScreenVM>
        {
            PauseResumeBackend BackendCommand;

            public override bool CanExecute(object? parameter)
            {
                return !parentViewModel.ObstacleHolder.UsingObstacleFile;
            }

            public override void Execute(object? parameter)
            {
                if (parentViewModel.EditingObstacles) // Obstacle editing is finished, need to embed obstacles and start backend executing.
                {
                    parentViewModel.EditingObstacles = false;
                    parentViewModel.EmbedObstacles();
                    BackendCommand.Execute(null);
                }
                else // Obstacle editing has started, need to stop backend and allow obstacles to be edited.
                {
                    BackendCommand.Execute(null);
                    parentViewModel.EditingObstacles = true;
                }
            }
            public EditObstacles(SimulationScreenVM parentViewModel) : base(parentViewModel)
            {
                BackendCommand = new PauseResumeBackend(parentViewModel);
            }
        }

        public class SelectObstacleFile : VMCommandBase<ConfigScreenVM>
        {
            public override bool CanExecute(object? parameter)
            {
                return parentViewModel.UsingObstacleFile;
            }
            public override void Execute(object? parameter)
            {
                OpenFileDialog openDialog = new();
                openDialog.Title = "Open Obstacle File";
                openDialog.Filter = "Obstacle Files (*.simobst)|*.simobst|Binary Files (*.bin)|*.bin|All Files|*.*";
                if (openDialog.ShowDialog() == true) // OK pressed rather than cancel
                {
                    parentViewModel.FileName = openDialog.FileName;
                }
            }

            public SelectObstacleFile(ConfigScreenVM parentViewModel) : base(parentViewModel)
            {
                parentViewModel.PropertyChanged += VMPropertyChanged;
            }

            private void VMPropertyChanged(object? sender, PropertyChangedEventArgs e)
            {
                if (e.PropertyName == nameof(parentViewModel.UsingObstacleFile))
                {
                    OnCanExecuteChanged(sender, e);
                }
            }
        }

        public class TrySimulate : VMCommandBase<ConfigScreenVM>
        {
            private ChangeWindow ChangeWindowCommand;
            public override void Execute(object? parameter)
            {
                if (parentViewModel.UsingObstacleFile && parentViewModel.FileName is null)
                {
                    MessageBox.Show("You must select a file with obstacles, or deselect the checkbox.", "ERROR: No file selected");
                }
                else
                {
                    ChangeWindowCommand.Execute(typeof(SimulationScreen));
                }
            }

            public TrySimulate(ConfigScreenVM parentViewModel) : base(parentViewModel)
            {
                ChangeWindowCommand = new();
            }
        }

        public class SimScreenBack : VMCommandBase<SimulationScreenVM>
        {
            private readonly PauseResumeBackend PauseResumeBackendCommand;
            private readonly ChangeWindow ChangeWindowCommand;
            public override bool CanExecute(object? parameter)
            {
                return !parentViewModel.EditingObstacles; // Cannot execute when editing obstacles.
            }

            public override void Execute(object? parameter)
            {
                if (parentViewModel.BackendStatus == BackendStatus.Running)
                {
                    PauseResumeBackendCommand.Execute(null);
                }
                Thread.Sleep(100);
                parentViewModel.CloseBackend();
                ChangeWindowCommand.Execute(typeof(ConfigScreen));
            }

            public SimScreenBack(SimulationScreenVM parentViewModel) : base(parentViewModel)
            {
                PauseResumeBackendCommand = new PauseResumeBackend(parentViewModel);
                ChangeWindowCommand = new ChangeWindow();
                parentViewModel.PropertyChanged += VMPropertyChanged;
            }

            private void VMPropertyChanged(object? sender, PropertyChangedEventArgs e)
            {
                if (e.PropertyName == nameof(parentViewModel.EditingObstacles)) OnCanExecuteChanged(sender, e);
            }
        }
    }
}
