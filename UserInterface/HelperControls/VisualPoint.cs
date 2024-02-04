using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace UserInterface.HelperControls
{
    public class VisualPoint : Shape
    {
        private const int defaultCircleRadiusRatio = 3;
        private const int hoverCircleRadiusRatio = 2;
        private const int size = 1;
        private const float aspectRatio = (float)9/16;
        private readonly DrawingBrush defaultFill;
        private readonly DrawingBrush mouseHoverFill;

        private Point point;
        private bool isDragged;

        protected override Geometry DefiningGeometry => new EllipseGeometry(new Point(0, 0), size * aspectRatio, size);

        public Point Point
        {
            get => point;
            set
            {
                point = value;
                Canvas.SetLeft(this, point.X - (size * aspectRatio) / 2);
                Canvas.SetBottom(this, point.Y - size);
            }
        }

        public bool IsDragged
        {
            get => isDragged;
            set
            {
                isDragged = value;
                OnIsDraggedChanged();
            }
        }

        public VisualPoint(Point point)
        {
            defaultFill = new DrawingBrush(new DrawingGroup
            {
                Children =
                [
                    new GeometryDrawing // Outer circle
                    {
                        Brush = Brushes.DarkGreen,
                        Geometry = new EllipseGeometry(new Point(0, 0), aspectRatio * defaultCircleRadiusRatio, defaultCircleRadiusRatio)
                    },
                    new GeometryDrawing // Inner, darker circle
                    {
                        Brush = Brushes.LightGreen,
                        Geometry = new EllipseGeometry(new Point(0, 0), aspectRatio, 1)
                    }
                ]
            });

            mouseHoverFill = new DrawingBrush(new DrawingGroup
            {
                Children =
                [
                    new GeometryDrawing // Outer circle
                    {
                        Brush = Brushes.DarkGreen,
                        Geometry = new EllipseGeometry(new Point(0, 0), aspectRatio * hoverCircleRadiusRatio, hoverCircleRadiusRatio)
                    },
                    new GeometryDrawing // Inner, darker circle
                    {
                        Brush = Brushes.LightGreen,
                        Geometry = new EllipseGeometry(new Point(0, 0), aspectRatio, 1)
                    }
                ]
            });

            isDragged = false;

            MouseEnter += OnMouseEnter;
            MouseLeave += OnMouseLeave;

            Fill = defaultFill;
            Point = point;
        }

        public VisualPoint() : this(new Point(0, 0)) { }

        public VisualPoint(double x, double y) : this(new Point(x, y)) { }

        private void OnIsDraggedChanged()
        {
            if (isDragged) // Dragging has just started
            {
                MouseEnter -= OnMouseEnter;
                MouseLeave -= OnMouseLeave;
                Fill = mouseHoverFill;
            }
            else // Dragging has just ended
            {
                MouseEnter += OnMouseEnter;
                MouseLeave += OnMouseLeave;
                Fill = defaultFill;
            }
        }

        private void OnMouseEnter(object sender, MouseEventArgs e) => Fill = mouseHoverFill;

        private void OnMouseLeave(object sender, MouseEventArgs e) => Fill = defaultFill;

        public override string ToString()
        {
            return point.ToString();
        }
    }
}
