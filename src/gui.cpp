#include <gtk/gtk.h>
#include <gdk/gdkcairo.h>
#include <cairo/cairo.h>
#include <iostream>
#include <vector>

struct Color
{
  static GdkRGBA black;
  static GdkRGBA white;
  static GdkRGBA grey;
  static GdkRGBA red;
  static GdkRGBA green;
  static GdkRGBA blue;
};

GdkRGBA Color::black  = {0.0, 0.0, 0.0, 1.0};
GdkRGBA Color::white  = {1.0, 1.0, 1.0, 1.0};
GdkRGBA Color::grey   = {0.5, 0.5, 0.5, 1.0};
GdkRGBA Color::red    = {1.0, 0.0, 0.0, 1.0};
GdkRGBA Color::green  = {0.0, 1.0, 0.0, 1.0};
GdkRGBA Color::blue   = {0.0, 0.0, 1.0, 1.0};

struct Point
{
  Point( double x, double y ) : x(x), y(y) {}
  double x;
  double y;
};

class Rectangle 
{
public:
  Rectangle( const Point& A, const Point& B ) 
    : drawColor( Color::red )
    , fillColor( Color::white )
    {
      minX = std::min( A.x ,B.x );
      maxX = std::max( A.x ,B.x );
      minY = std::min( A.y ,B.y );
      maxY = std::max( A.y ,B.y );
    }

  Rectangle( const Rectangle& other ) 
    : minX( other.minX )
      , maxX( other.maxX )
      , minY( other.minY )
      , maxY( other.maxY )
      , drawColor( other.drawColor )
      , fillColor( other.fillColor )
    { }

  void drawSelf( cairo_t* cr )
    {

      cairo_set_source_rgb( cr, fillColor.red, fillColor.green, fillColor.blue  );
      cairo_set_line_width( cr, 1 );

      cairo_rectangle( cr, minX, minY, maxX, maxY );
      cairo_fill_preserve( cr );
      cairo_set_source_rgb( cr, drawColor.red, drawColor.green, drawColor.blue  );
      cairo_stroke( cr );

    }

  void inverseFillColor()
    {
      fillColor.red   = 1.0 - fillColor.red;
      fillColor.green = 1.0 - fillColor.green;
      fillColor.blue  = 1.0 - fillColor.blue;
    }

  bool is_inside( const Point& p )
    {
      return ( minX <= p.x ) && ( p.x <= maxX ) && ( minY <= p.y ) && ( p.y <= maxY );
    }

  friend std::ostream& operator<<( std::ostream& os, const Rectangle& r );

protected:
  int minX, maxX, minY, maxY;
  GdkRGBA drawColor;
  GdkRGBA fillColor;
};

std::ostream& operator<<( std::ostream& os, const Rectangle& r )
{
  os << "Rectangle( (" << r.minX << ", " << r.minY << "), (" << r.maxX << ", " << r.maxY << ") )";
  return os;
}

     

gboolean
draw_callback (GtkWidget *widget, cairo_t *cr, gpointer data)
{
  std::vector<Rectangle>* v = (std::vector<Rectangle>*) data;

  static double d = 2.0;
  static double inc = -0.1;


  guint width, height;
  GdkRGBA color;
  GtkStyleContext *context;

  context = gtk_widget_get_style_context (widget);

  width = gtk_widget_get_allocated_width (widget);
  height = gtk_widget_get_allocated_height (widget);

  gtk_render_background (context, cr, 0, 0, width, height);

  cairo_arc (cr,
             width / 2.0, height / 2.0,
             MIN (width, height) / 2.0,
             0, d * G_PI);
  d += inc;
  if ( d < 0.0 || d > 2.0)
    {
      inc = -inc;
      d += 2*inc;
    }



  gtk_style_context_get_color (context,
                               gtk_style_context_get_state (context),
                               &color);
  gdk_cairo_set_source_rgba (cr, &color);

  cairo_fill (cr);

  for ( Rectangle r : *v ) 
    {
      r.drawSelf( cr );
    }


 return FALSE;
}

static gboolean 
clicked(GtkWidget *widget, GdkEventButton *event, gpointer user_data)
{
  std::vector<Rectangle>* v = (std::vector<Rectangle>*) user_data;
  Point p( (int)event->x, (int)event->y );
  std::cout << "widget " << widget << std::endl;
  std::cout << "click ("  << event->x << ", " << event->y << ")"<< std::endl;
  if ( event->button == 1 )
    {
      for ( Rectangle& r : *v )
        {
          if ( r.is_inside( p ) )
            {
              std::cout << "Inside " << r << std::endl;
              r.inverseFillColor();
            }
          else
            {
              std::cout << "Outside " << r << std::endl;
            }
        }
      gtk_widget_queue_draw(widget);
    }
  //draw_callback( widget, (cairo_t*) widget, NULL );

}

static void activate (GtkApplication *app, gpointer user_data)
{
  std::vector<Rectangle>* v = (std::vector<Rectangle>*) user_data;
  GtkWidget *window;
  GtkWidget *drawing_area;

  /* create a new window, and set its title */
  window = gtk_application_window_new (app);
  gtk_window_set_title (GTK_WINDOW (window), "Now I understand !");
  gtk_container_set_border_width (GTK_CONTAINER (window), 10);

  /* Here we construct the container that is going pack our buttons */
  drawing_area = gtk_drawing_area_new();
  gtk_widget_set_size_request (drawing_area, 700, 500);
  g_signal_connect (G_OBJECT (drawing_area), "draw",
                    G_CALLBACK (draw_callback), (gpointer) v );
  g_signal_connect(G_OBJECT(window), "button-press-event", G_CALLBACK(clicked), (gpointer) v);

  /* Pack the container in the window */
  gtk_container_add (GTK_CONTAINER (window), drawing_area);

  gtk_widget_show_all (window);

}

int main (int argc, char **argv)
{
  GtkApplication *app;
  int status;

  app = gtk_application_new ("org.gtk.example", G_APPLICATION_FLAGS_NONE);

  std::vector<Rectangle> v; 
  v.push_back( Rectangle( Point(10, 10), Point(50, 50) ) );
  v.push_back( Rectangle( Point(310, 310), Point(350, 350) ) );

  g_signal_connect (app, "activate", G_CALLBACK (activate), (gpointer) &v );
  status = g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);

  return status;
}
