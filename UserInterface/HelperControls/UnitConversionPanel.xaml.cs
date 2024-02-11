using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using UserInterface.Converters;
using UserInterface.HelperClasses;

namespace UserInterface.HelperControls
{
    /// <summary>
    /// Interaction logic for UnitConversionPanel.xaml
    /// </summary>
    public partial class UnitConversionPanel : UserControl
    {
        private readonly UnitHolder unitHolder;

        public UnitConversionPanel(UnitHolder unitHolder)
        {
            InitializeComponent();
            this.unitHolder = unitHolder;
            SetInitialComboBoxContents();
            unitHolder.PropertyChanged += OnUnitChanged;
        }

        private void SetInitialComboBoxContents()
        {
            SetComboBoxContents(unitHolder.LengthUnit, LengthComboBox);
            SetComboBoxContents(unitHolder.SpeedUnit, SpeedComboBox);
            SetComboBoxContents(unitHolder.TimeUnit, TimeComboBox);
            SetComboBoxContents(unitHolder.DensityUnit, DensityComboBox);
            SetComboBoxContents(unitHolder.ViscosityUnit, ViscosityComboBox);
        }

        private void OnUnitChanged(object? sender, PropertyChangedEventArgs e)
        {
            UnitChangedEventArgs args = e as UnitChangedEventArgs;
            ComboBox? relevantComboBox = args.PropertyName switch
            {
                nameof(unitHolder.LengthUnit) => LengthComboBox,
                nameof(unitHolder.SpeedUnit) => SpeedComboBox,
                nameof(unitHolder.TimeUnit) => TimeComboBox,
                nameof(unitHolder.DensityUnit) => DensityComboBox,
                nameof(unitHolder.ViscosityUnit) => ViscosityComboBox,
                _ => null
            };
            if (relevantComboBox is null) return;
            SetComboBoxContents(args.NewValue, relevantComboBox);
        }

        private void SetComboBoxContents(UnitClasses.Unit newUnit, ComboBox? relevantComboBox)
        {
            relevantComboBox.SelectionChanged -= OnSelectionChanged;

            foreach (var item in relevantComboBox.Items)
            {
                if (item is ComboBoxItem comboBoxItem && (string)comboBoxItem.Content == newUnit.LongName)
                {
                    comboBoxItem.IsSelected = true;
                }
            }
            relevantComboBox.SelectionChanged += OnSelectionChanged;
        }

        private void OnUnitSystemChanged(object sender, SelectionChangedEventArgs e)
        {
            int selectedIndex = (sender as ComboBox).SelectedIndex;
            if (selectedIndex == -1 || selectedIndex > 2) return;
            UnitClasses.UnitSystem unitSystem = (UnitClasses.UnitSystem)selectedIndex;
            unitHolder.SetUnitSystem(unitSystem);
        }

        private void OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (sender is not ComboBox relevantComboBox) return;
            UnitSystemComboBox.SelectedIndex = -1;
            unitHolder.PropertyChanged -= OnUnitChanged;
            switch (relevantComboBox.Name)
            {
                case "LengthComboBox":
                    unitHolder.LengthUnit = relevantComboBox.SelectedIndex switch
                    {
                        0 => new UnitClasses.Metre(),
                        1 => new UnitClasses.Foot(),
                        2 => new UnitClasses.Inch(),
                        3 => new UnitClasses.Centimetre(),
                        4 => new UnitClasses.Millimetre(),
                        _ => null
                    };
                    break;
                case "SpeedComboBox":
                    unitHolder.SpeedUnit = relevantComboBox.SelectedIndex switch
                    {
                        0 => new UnitClasses.MetrePerSecond(),
                        1 => new UnitClasses.CentimetrePerSecond(),
                        2 => new UnitClasses.MilePerHour(),
                        3 => new UnitClasses.KilometrePerHour(),
                        4 => new UnitClasses.FootPerSecond(),
                        _ => null
                    };
                    break;
                case "TimeComboBox":
                    unitHolder.TimeUnit = relevantComboBox.SelectedIndex switch
                    {
                        0 => new UnitClasses.Second(),
                        1 => new UnitClasses.Millisecond(),
                        2 => new UnitClasses.Minute(),
                        3 => new UnitClasses.Hour(),
                        _ => null
                    };
                    break;
                case "DensityComboBox":
                    unitHolder.DensityUnit = relevantComboBox.SelectedIndex switch
                    {
                        0 => new UnitClasses.KilogramPerCubicMetre(),
                        1 => new UnitClasses.GramPerCubicCentimetre(),
                        2 => new UnitClasses.PoundPerCubicInch(),
                        3 => new UnitClasses.PoundPerCubicFoot(),
                        _ => null
                    };
                    break;
                case "ViscosityComboBox":
                    unitHolder.ViscosityUnit = relevantComboBox.SelectedIndex switch
                    {
                        0 => new UnitClasses.KilogramPerMetrePerSecond(),
                        1 => new UnitClasses.GramPerCentimetrePerSecond(),
                        2 => new UnitClasses.PoundSecondPerSquareFoot(),
                        _ => null
                    };
                    break;
            }
            unitHolder.PropertyChanged += OnUnitChanged;
        }
    }
}
