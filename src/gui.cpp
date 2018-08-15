#include <gtk/gtk.h>
#include <gdk/gdkcairo.h>
#include <iostream>


static void print_hello (GtkWidget *widget, gpointer   data)
{
  (void) widget;
  (void) data;
  g_print ("Hello World\n");
}

double d = 2.0;
double inc = -0.1;

gboolean
draw_callback (GtkWidget *widget, cairo_t *cr, gpointer data)
{
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

 return FALSE;
}

static gboolean 
clicked(GtkWidget *widget, GdkEventButton *event, gpointer user_data)
{
  (void) user_data;
  std::cout << "click ("  << event->x << ", " << event->y << ")"<< std::endl;
  if ( event->button == 1 )
    {
      gtk_widget_queue_draw(widget);
    }
  //draw_callback( widget, (cairo_t*) widget, NULL );

}

static void activate (GtkApplication *app, gpointer user_data)
{
  (void) user_data;
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
                    G_CALLBACK (draw_callback), NULL);
  g_signal_connect(G_OBJECT(window), "button-press-event", G_CALLBACK(clicked), NULL);
  //grid = gtk_grid_new ();

  /* Pack the container in the window */
  gtk_container_add (GTK_CONTAINER (window), drawing_area);

  gtk_widget_show_all (window);

}

int main (int argc, char **argv)
{
  GtkApplication *app;
  int status;

  app = gtk_application_new ("org.gtk.example", G_APPLICATION_FLAGS_NONE);
  g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);
  status = g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);

  return status;
}
