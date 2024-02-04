using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using UserInterface.Converters;
using UserInterface.HelperClasses;
using UserInterface.HelperControls;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : UserControl
    {
        private readonly SimulationScreenVM viewModel;
        private readonly RectangularToPolar RecToPolConverter;
        public SimulationScreenVM ViewModel => viewModel;

        private VisualPoint? draggedPoint;
        private bool isCentreMoved;
        private Point mousePosition;
        private bool pointsPlaced;

        private static readonly double POINT_TOLERANCE = 0.1f;

        public SimulationScreen(ParameterHolder parameterHolder)
        {
            InitializeComponent();

            viewModel = new SimulationScreenVM(parameterHolder);
            DataContext = viewModel;

            RecToPolConverter = new();
            pointsPlaced = false;
            isCentreMoved = false;
            ViewModel.PropertyChanged += OnViewModelPropertyChanged;
        }

        #region Private helper methods
        private void PlaceInitialPoints()
        {
            foreach (PolarPoint polarPoint in ViewModel.ControlPoints)
            {
                SimulationCanvas.Children.Add(new VisualPoint(ConvertToRect(polarPoint)));
            }
            SimulationCanvas.Children.Add(new VisualPoint(ViewModel.ObstacleCentre));
        }

        private void MoveControlPoints(Vector translation)
        {
            for (int i = 0; i < SimulationCanvas.Children.Count; i++)
            {
                if (SimulationCanvas.Children[i] is VisualPoint point)
                {
                    point.Point += translation;
                }
            }
        }

        private void AddPoint(VisualPoint point)
        {
            SimulationCanvas.Children.Add(point);
            ViewModel.ControlPoints.Add(ConvertToPolar(point.Point));
        }

        private void RemovePoint(VisualPoint point)
        {
            SimulationCanvas.Children.Remove(point);
            PolarPoint polarPoint = ConvertToPolar(point.Point);
            int polarPointIndex = FindIndexOfPolarPoint(polarPoint);
            ViewModel.ControlPoints.RemoveAt(polarPointIndex);
        }

        private int FindIndexOfPolarPoint(PolarPoint polarPoint)
        {
            int polarPointIndex = ViewModel.ControlPoints.IndexOf(polarPoint);
            if (polarPointIndex == -1)
            {
                for (int i = 0; i < ViewModel.ControlPoints.Count; i++)
                {
                    PolarPoint comparisonPoint = ViewModel.ControlPoints[i];
                    if (Math.Abs(polarPoint.Radius - comparisonPoint.Radius) < POINT_TOLERANCE && Math.Abs(polarPoint.Angle - comparisonPoint.Angle) < POINT_TOLERANCE) // Allow some inexactness
                    {
                        polarPointIndex = i;
                    }
                }
                if (polarPointIndex == -1) // If still not found
                {
                    throw new InvalidOperationException("Could not find index of polar point.");
                }
            }

            return polarPointIndex;
        }

        private PolarPoint ConvertToPolar(Point rectangularPoint)
        {
            return (PolarPoint)RecToPolConverter.Convert(rectangularPoint, typeof(PolarPoint), ViewModel.ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture);
        }

        private Point ConvertToRect(PolarPoint polarPoint)
        {
            return (Point)RecToPolConverter.ConvertBack(polarPoint, typeof(PolarPoint), ViewModel.ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture);
        }
        #endregion

        #region Event handlers
        private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            switch (e.PropertyName)
            {
                case nameof(ViewModel.EditingObstacles):
                    OnEditingObstalesChanged();
                    break;
                default:
                    break;
            }
        }

        private void OnEditingObstalesChanged()
        {
            if (ViewModel.EditingObstacles && !pointsPlaced) // Place the points the first time app enters obstacle editing.
            {
                PlaceInitialPoints();
                pointsPlaced = true;
            }

            foreach (UIElement element in SimulationCanvas.Children) // Set the visibility of all of the control points on entering/leaving obstacle editing
            {
                if (element is VisualPoint)
                {
                    element.Visibility = ViewModel.EditingObstacles ? Visibility.Visible : Visibility.Hidden;
                }
            }
        }

        private void CanvasMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && SimulationCanvas.CaptureMouse())
            {
                mousePosition = e.GetPosition(SimulationCanvas);
                draggedPoint = point;
                draggedPoint.IsDragged = true;
                Trace.WriteLine($"difference in X coordinate: {Math.Abs(draggedPoint.Point.X - ViewModel.ObstacleCentre.X)}");
                Trace.WriteLine($"difference in Y coordinate: {Math.Abs(draggedPoint.Point.Y - ViewModel.ObstacleCentre.Y)}");
                isCentreMoved = Math.Abs(draggedPoint.Point.X - ViewModel.ObstacleCentre.X) < POINT_TOLERANCE && Math.Abs(draggedPoint.Point.Y - ViewModel.ObstacleCentre.Y) < POINT_TOLERANCE;
                
                Panel.SetZIndex(draggedPoint, 1); // Make point go in front of everything else while is is dragged
            }
            else
            {
                Point clickPosition = e.GetPosition(SimulationCanvas); // Check whether mouse positions map correctly or not.
                AddPoint(new VisualPoint(clickPosition.X, 100 - clickPosition.Y));
            }
        }

        private void CanvasMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (draggedPoint is not null)
            {
                SimulationCanvas.ReleaseMouseCapture();
                Panel.SetZIndex(draggedPoint, 0);
                draggedPoint.IsDragged = false;
                draggedPoint = null;
                isCentreMoved = false;
            }
        }

        private void CanvasMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && point.Point != ViewModel.ObstacleCentre)
            {
                RemovePoint(point);
            }
        }

        private void CanvasMouseMove(object sender, MouseEventArgs e)
        {
            if (draggedPoint != null)
            {
                Point position = e.GetPosition(SimulationCanvas);
                Vector offset = position - mousePosition;
                mousePosition = position;
                Vector offsetYFlip = new Vector(offset.X, -offset.Y);
                if (!isCentreMoved) // Normal control points
                {
                    int draggedPointIndex = FindIndexOfPolarPoint(ConvertToPolar(draggedPoint.Point));
                    draggedPoint.Point += offsetYFlip;
                    ViewModel.ControlPoints[draggedPointIndex] = ConvertToPolar(draggedPoint.Point);
                }
                else
                {
                    MoveControlPoints(offsetYFlip);

                    ViewModel.ObstacleCentre = draggedPoint.Point;
                }
            }
        }
        #endregion
    }
}
