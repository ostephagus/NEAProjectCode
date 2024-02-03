using System;
using System.ComponentModel;
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
    public partial class SimulationScreen : SwappableScreen
    {
        private readonly SimulationScreenVM viewModel;
        private readonly RelativeDimension coordinateConverter;
        private readonly AbsoluteRectToRelativePol RecToPolConverter;
        private readonly RelativePolToAbsoluteRect PolToRecConverter;
        public SimulationScreenVM ViewModel => viewModel;

        private VisualPoint? centrePoint;
        private VisualPoint? draggedPoint;
        private Point mousePosition;
        private Point dragStartPoint;
        private bool pointsPlaced;

        private static readonly double pointTolerance = 0.1f;

        public SimulationScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            viewModel = new SimulationScreenVM(parameterHolder);
            DataContext = viewModel;

            coordinateConverter = new();
            RecToPolConverter = new();
            PolToRecConverter = new();
            pointsPlaced = false;
            ViewModel.PropertyChanged += OnViewModelPropertyChanged;
        }

        private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            switch (e.PropertyName)
            {
                case nameof(ViewModel.EditingObstacles):
                    OnEditingObstalesChanged();
                    break;
                case nameof(ViewModel.ObstacleCentre):
                    OnObstacleCentreChanged();
                    break;
                default:
                    break;
            }
        }

        private void OnObstacleCentreChanged()
        {
            MoveControlPoints();
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

        private void PlaceInitialPoints()
        {
            foreach (PolarPoint polarPoint in ViewModel.ControlPoints)
            {
                SimulationCanvas.Children.Add(new VisualPoint((Point)PolToRecConverter.Convert([polarPoint, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight], typeof(Point), ViewModel.ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture)));
            }
            centrePoint = new VisualPoint(new Point((double)coordinateConverter.Convert(SimulationCanvas.ActualWidth, typeof(Point), (ViewModel.ObstacleCentre.X / 2).ToString(), System.Globalization.CultureInfo.CurrentCulture),
                                                    (double)coordinateConverter.Convert(SimulationCanvas.ActualHeight, typeof(Point), (ViewModel.ObstacleCentre.Y / 2).ToString(), System.Globalization.CultureInfo.CurrentCulture)));
            SimulationCanvas.Children.Add(centrePoint);
        }

        private void MoveControlPoints()
        {
            int pointNumber = 0;
            for (int i = 0; i < SimulationCanvas.Children.Count; i++)
            {
                if (SimulationCanvas.Children[i] is VisualPoint point && point != centrePoint)
                {
                    point.Point = (Point)PolToRecConverter.Convert([ViewModel.ObstaclePoints[pointNumber], SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight], typeof(Point), ViewModel.ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture);
                    SimulationCanvas.Children[i] = point;
                    pointNumber++;
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

        private void CanvasMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && SimulationCanvas.CaptureMouse())
            {
                mousePosition = e.GetPosition(SimulationCanvas);
                dragStartPoint = mousePosition;
                draggedPoint = point;
                
                Panel.SetZIndex(draggedPoint, 1); // Make point go in front of everything else while is is dragged
            }
            else
            {
                Point clickPosition = e.GetPosition(SimulationCanvas);
                AddPoint(new VisualPoint(new Point(clickPosition.X / 2, clickPosition.Y / 2))); // Halve the point position.
            }
        }

        private void CanvasMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (draggedPoint is null) return;

            SimulationCanvas.ReleaseMouseCapture();
            Panel.SetZIndex(draggedPoint, 0);
            Vector dragDisplacement = e.GetPosition(SimulationCanvas) - dragStartPoint;
            Point dragStartPointHalved = new Point(dragStartPoint.X / 2, dragStartPoint.Y / 2);

            if (draggedPoint == centrePoint)
            {
                Point dragEndPoint = dragStartPointHalved + dragDisplacement;
                centrePoint.Point = dragEndPoint;
                ViewModel.ObstacleCentre = new Point((double)coordinateConverter.ConvertBack(SimulationCanvas.ActualWidth, typeof(Point), (dragEndPoint.X * 2).ToString(), System.Globalization.CultureInfo.CurrentCulture),
                                                    1 - (double)coordinateConverter.ConvertBack(SimulationCanvas.ActualHeight, typeof(Point), (dragEndPoint.Y * 2).ToString(), System.Globalization.CultureInfo.CurrentCulture));
            }
            else // DraggedPoint is a control point
            {

                Point dragEndPoint = dragStartPointHalved + (dragDisplacement * 0.5);
                PolarPoint startPolarPoint = ConvertToPolar(dragStartPointHalved);
                int draggedPointIndex = FindIndexOfPolarPoint(startPolarPoint);

                ViewModel.ControlPoints[draggedPointIndex] = ConvertToPolar(dragEndPoint);
            }

            draggedPoint = null;
        }

        private int FindIndexOfPolarPoint(PolarPoint polarPoint)
        {
            int polarPointIndex = ViewModel.ControlPoints.IndexOf(polarPoint);
            if (polarPointIndex == -1)
            {
                for (int i = 0; i < ViewModel.ControlPoints.Count; i++)
                {
                    PolarPoint comparisonPoint = ViewModel.ControlPoints[i];
                    if (Math.Abs(polarPoint.Radius - comparisonPoint.Radius) < pointTolerance && Math.Abs(polarPoint.Angle - comparisonPoint.Angle) < pointTolerance) // Allow some inexactness
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

        /// <summary>
        /// Fills in the common parameters and calls <see cref="AbsoluteRectToRelativePol.Convert(object[], Type, object, System.Globalization.CultureInfo)"/>, like a partial function application.
        /// </summary>
        /// <param name="rectangularPoint">The absolute rectangular point to convert</param>
        /// <returns>A relative polar coordinate.</returns>
        private PolarPoint ConvertToPolar(Point rectangularPoint)
        {
            return (PolarPoint)RecToPolConverter.Convert([rectangularPoint, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight], typeof(PolarPoint), ViewModel.ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture);
        }

        private void CanvasMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && point != centrePoint)
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
                //PolarPoint oldPoint = (PolarPoint)RecToPolConverter.Convert(new object[] { draggedPoint.Point, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight }, typeof(PolarPoint), new Point(0.5, 0.5), System.Globalization.CultureInfo.CurrentCulture);
                draggedPoint.Point += offset;
                //PolarPoint newPoint = (PolarPoint)RecToPolConverter.Convert(new object[] { draggedPoint.Point, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight }, typeof(PolarPoint), new Point(0.5, 0.5), System.Globalization.CultureInfo.CurrentCulture);
                //ViewModel.ControlPoints[ViewModel.ControlPoints.IndexOf(oldPoint)] = newPoint; // Replace the point.
            }
        }
    }
}
