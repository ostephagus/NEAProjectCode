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

        public float Tau 
        { 
            get => tau; 
            set { tau = value; OnPropertyChanged(this, nameof(Tau)); } 
        }

        public float Omega { 
            get => omega; 
            set { omega = value; OnPropertyChanged(this, nameof(Omega)); }
        }

        public float RMax
        {
            get => rMax;
            set { rMax = value; OnPropertyChanged(this, nameof(RMax)); }
        }

        public float IterMax
        {
            get => iterMax;
            set { iterMax = value; OnPropertyChanged(this, nameof(IterMax)); }
        }

        public Commands.AdvancedParametersReset ResetCommand { get; set; }

        public Commands.SaveCommand SaveCommand { get; set; }
        #endregion

        public AdvancedParametersVM(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            Tau = parameterHolder.TimeStepSafetyFactor.Value;
            Omega = parameterHolder.RelaxationParameter.Value;
            RMax = parameterHolder.PressureResidualTolerance.Value;
            IterMax = parameterHolder.PressureMaxIterations.Value;

            ResetCommand = new Commands.AdvancedParametersReset(this, parameterHolder);
            SaveCommand = new Commands.SaveCommand(this, parameterHolder, new Commands.ChangeWindow());
        }
    }
}
