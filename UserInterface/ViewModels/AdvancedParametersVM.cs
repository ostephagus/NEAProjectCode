using System.Windows.Navigation;
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

        public float DelX
        {
            get => parameterHolder.Width.Value / obstacleHolder.DataWidth;
            set
            {
                obstacleHolder.DataWidth = (int)(parameterHolder.Width.Value / value);
                OnPropertyChanged(this, nameof(DelX));
            }
        }
        public float DelY
        {
            get => parameterHolder.Height.Value / obstacleHolder.DataHeight;
            set
            {
                obstacleHolder.DataHeight = (int)(parameterHolder.Height.Value / value);
                OnPropertyChanged(this, nameof(DelY));
            }
        }

        public bool CanChangeGridSizes
        {
            get => !obstacleHolder.UsingObstacleFile;
        }

        public Commands.AdvancedParametersReset ResetCommand { get; set; }

        public Commands.SaveParameters SaveCommand { get; set; }
        #endregion

        public AdvancedParametersVM(ParameterHolder parameterHolder, UnitHolder unitHolder, ObstacleHolder obstacleHolder) : base(parameterHolder, unitHolder, obstacleHolder)
        {
            Tau = parameterHolder.TimeStepSafetyFactor.Value;
            Omega = parameterHolder.RelaxationParameter.Value;
            RMax = parameterHolder.PressureResidualTolerance.Value;
            IterMax = parameterHolder.PressureMaxIterations.Value;

            ResetCommand = new Commands.AdvancedParametersReset(this, parameterHolder);
            SaveCommand = new Commands.SaveParameters(this, parameterHolder, new Commands.ChangeWindow());
        }
    }
}
