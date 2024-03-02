using UserInterface.HelperClasses;
using UserInterface.HelperControls;

namespace UserInterface.ViewModels
{
    public class ConfigScreenVM : ViewModel
    {
        private readonly UnitConversionPanel unitsPanel;

        private string? fileName;
        private bool usingObstacleFile;

        public float InVel
        {
            get => parameterHolder.InflowVelocity.Value;
            set
            {
                parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, value);
                OnPropertyChanged(this, nameof(InVel));
            }
        }
        public float Chi
        {
            get => parameterHolder.SurfaceFriction.Value;
            set
            {
                parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, value);
                OnPropertyChanged(this, nameof(Chi));
            }
        }
        public float Width
        {
            get => parameterHolder.Width.Value;
            set
            {
                parameterHolder.Width = ModifyParameterValue(parameterHolder.Width, value);
                OnPropertyChanged(this, nameof(Width));
            }
        }
        public float Height
        {
            get => parameterHolder.Height.Value;
            set
            {
                parameterHolder.Height = ModifyParameterValue(parameterHolder.Height, value);
                OnPropertyChanged(this, nameof(Height));
            }
        }
        public float ReynoldsNo
        {
            get => parameterHolder.ReynoldsNumber.Value;
            set
            {
                parameterHolder.ReynoldsNumber = ModifyParameterValue(parameterHolder.ReynoldsNumber, value);
                OnPropertyChanged(this, nameof(ReynoldsNo));
            }
        }
        public float Viscosity
        {
            get => parameterHolder.FluidViscosity.Value;
            set {
                parameterHolder.FluidViscosity = ModifyParameterValue(parameterHolder.FluidViscosity, value);
                OnPropertyChanged(this, nameof(Viscosity));
            }
        }
        public float Density
        {
            get => parameterHolder.FluidDensity.Value;
            set
            {
                parameterHolder.FluidDensity = ModifyParameterValue(parameterHolder.FluidDensity, value);
                OnPropertyChanged(this, nameof(Density));
            }
        }

        public string? FileName
        {
            get => fileName;
            set
            {
                fileName = value;
                OnPropertyChanged(this, nameof(FileName));
                OnPropertyChanged(this, nameof(DisplayFileName));
            }
        }
        public string DisplayFileName
        {
            get
            {
                if (UsingObstacleFile == false)
                {
                    return "Not using obstacle files.";
                }
                if (fileName == null)
                {
                    return "No File Selected.";
                }
                string[] fileParts = fileName.Split('\\');
                return $"File selected: {fileParts[^1]}";
            }
        }
        public bool UsingObstacleFile
        {
            get => usingObstacleFile;
            set
            {
                usingObstacleFile = value;
                OnPropertyChanged(this, nameof(UsingObstacleFile));
                OnPropertyChanged(this, nameof(DisplayFileName));
            }
        }

        public UnitConversionPanel UnitsPanel => unitsPanel;

        public Commands.ConfigScreenReset ResetCommand { get; set; }
        public Commands.SetAirParameters SetAirCommand { get; set; }
        public Commands.SelectObstacleFile SelectObstacleFileCommand { get; set; }
        public Commands.TrySimulate TrySimulateCommand { get; set; }
        public Commands.CreatePopup CreatePopupCommand { get; set; }

        public ConfigScreenVM(ParameterHolder parameterHolder, UnitHolder unitHolder) : base(parameterHolder, unitHolder)
        {
            InVel = parameterHolder.InflowVelocity.Value;
            Chi = parameterHolder.SurfaceFriction.Value;
            Width = parameterHolder.Width.Value;
            Height = parameterHolder.Height.Value;
            ReynoldsNo = parameterHolder.ReynoldsNumber.Value;
            Viscosity = parameterHolder.FluidViscosity.Value;
            Density = parameterHolder.FluidDensity.Value;

            usingObstacleFile = true;

            unitsPanel = new UnitConversionPanel(unitHolder);
            ResetCommand = new Commands.ConfigScreenReset(this, parameterHolder);
            SetAirCommand = new Commands.SetAirParameters(this);
            SelectObstacleFileCommand = new Commands.SelectObstacleFile(this);
            TrySimulateCommand = new Commands.TrySimulate(this);
            CreatePopupCommand = new Commands.CreatePopup();
        }
    }
}
