using System.ComponentModel;
using UserInterface.HelperClasses;

namespace UserInterface.ViewModels
{
    internal class AdvancedParametersVM : INotifyPropertyChanged
    {
        private float tau;
        private float omega;
        private float rMax;
        private float iterMax;

        public float Tau { 
            get => tau; 
            set
            {
                tau = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Tau)));
            } 
        }

        public float Omega { 
            get => omega; 
            set
            {
                omega = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Omega)));
            }
        }

        public float RMax
        {
            get => rMax;
            set
            {
                rMax = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(RMax)));
            }
        }

        public float IterMax
        {
            get => iterMax;
            set
            {
                iterMax = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IterMax)));
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        public AdvancedParametersVM()
        {
            //Tau = parameterHolder.TimeStepSafetyFactor.Value;
            //Omega = parameterHolder.RelaxationParameter.Value;
            //RMax = parameterHolder.PressureResidualTolerance.Value;
            //IterMax = parameterHolder.PressureMaxIterations.Value;
        }
    }
}
