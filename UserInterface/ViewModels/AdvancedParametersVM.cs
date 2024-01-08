using System;
using System.Diagnostics;
using System.Windows.Input;
using UserInterface.HelperClasses;

namespace UserInterface.ViewModels
{
    public class AdvancedParametersVM : ViewModel
    {
        #region Field and Properties
        private float tau;
        private float omega;
        private float rMax;
        private float iterMax;

        public float Tau { 
            get => tau; 
            set
            {
                tau = value;
                OnPropertyChanged(this, nameof(Tau));
                Trace.WriteLine("Tau changed");
            } 
        }

        public float Omega { 
            get => omega; 
            set
            {
                omega = value;
                OnPropertyChanged(this, nameof(Omega));
            }
        }

        public float RMax
        {
            get => rMax;
            set
            {
                rMax = value;
                OnPropertyChanged(this, nameof(RMax));
            }
        }

        public float IterMax
        {
            get => iterMax;
            set
            {
                iterMax = value;
                OnPropertyChanged(this, nameof(IterMax));
            }
        }

        public Commands.ResetCommand ResetCommand { get; set; }
        #endregion

        public AdvancedParametersVM(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            Tau = parameterHolder.TimeStepSafetyFactor.Value;
            Omega = parameterHolder.RelaxationParameter.Value;
            RMax = parameterHolder.PressureResidualTolerance.Value;
            IterMax = parameterHolder.PressureMaxIterations.Value;
            ResetCommand = new Commands.ResetCommand(this, parameterHolder);
        }
    }
}
