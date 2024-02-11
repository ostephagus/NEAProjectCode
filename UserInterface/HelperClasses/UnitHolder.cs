using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using UserInterface.Converters;

namespace UserInterface.HelperClasses
{
    public class UnitHolder : INotifyPropertyChanged
    {
        private UnitClasses.LengthUnit lengthUnit;
        private UnitClasses.SpeedUnit speedUnit;
        private UnitClasses.TimeUnit timeUnit;
        private UnitClasses.DensityUnit densityUnit;
        private UnitClasses.ViscosityUnit viscosityUnit;

        private readonly UnitClasses.UnitSystemList SIList;
        private readonly UnitClasses.UnitSystemList CGSList;
        private readonly UnitClasses.UnitSystemList DefaultImperialList;

        public UnitClasses.LengthUnit LengthUnit
        {
            get => lengthUnit;
            set
            {
                lengthUnit = value;
                OnPropertyChanged(LengthUnit);
            }
        }
        public UnitClasses.SpeedUnit SpeedUnit
        {
            get => speedUnit;
            set
            {
                speedUnit = value;
                OnPropertyChanged(SpeedUnit);
            }
        }
        public UnitClasses.TimeUnit TimeUnit
        {
            get => timeUnit;
            set
            {
                timeUnit = value;
                OnPropertyChanged(TimeUnit);
            }
        }
        public UnitClasses.DensityUnit DensityUnit
        {
            get => densityUnit;
            set
            {
                densityUnit = value;
                OnPropertyChanged(DensityUnit);
            }
        }
        public UnitClasses.ViscosityUnit ViscosityUnit
        {
            get => viscosityUnit;
            set
            {
                viscosityUnit = value;
                OnPropertyChanged(ViscosityUnit);
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        public UnitHolder()
        {
            lengthUnit = new UnitClasses.Metre();
            speedUnit = new UnitClasses.MetrePerSecond();
            timeUnit = new UnitClasses.Second();
            densityUnit = new UnitClasses.KilogramPerCubicMetre();
            viscosityUnit = new UnitClasses.KilogramPerMetrePerSecond();

            SIList = new UnitClasses.UnitSystemList(UnitClasses.UnitSystem.SI, new UnitClasses.Metre(), new UnitClasses.MetrePerSecond(), new UnitClasses.Second(), new UnitClasses.KilogramPerCubicMetre(), new UnitClasses.KilogramPerMetrePerSecond());
            CGSList = new UnitClasses.UnitSystemList(UnitClasses.UnitSystem.CGS, new UnitClasses.Centimetre(), new UnitClasses.CentimetrePerSecond(), new UnitClasses.Second(), new UnitClasses.GramPerCubicCentimetre(), new UnitClasses.GramPerCentimetrePerSecond());
            DefaultImperialList = new UnitClasses.UnitSystemList(UnitClasses.UnitSystem.SI, new UnitClasses.Foot(), new UnitClasses.FootPerSecond(), new UnitClasses.Second(), new UnitClasses.PoundPerCubicFoot(), new UnitClasses.PoundSecondPerSquareFoot());
        }

        private void OnPropertyChanged(UnitClasses.Unit value, [CallerMemberName] string name = "")
        {
            PropertyChanged?.Invoke(this, new UnitChangedEventArgs(name, value));
        }

        public void SetUnitSystem(UnitClasses.UnitSystem unitSystem)
        {
            UnitClasses.UnitSystemList unitSystemList = unitSystem switch
            {
                UnitClasses.UnitSystem.SI => SIList,
                UnitClasses.UnitSystem.CGS => CGSList,
                UnitClasses.UnitSystem.DefaultImperial => DefaultImperialList,
                _ => throw new ArgumentException("Unit system parameter was not a valid unit system.", nameof(unitSystem))
            };

            LengthUnit = unitSystemList.lengthUnit;
            SpeedUnit = unitSystemList.speedUnit;
            TimeUnit = unitSystemList.timeUnit;
            DensityUnit = unitSystemList.densityUnit;
            ViscosityUnit = unitSystemList.viscosityUnit;
        }
    }
}
