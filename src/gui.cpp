#include <stdlib.h>
#include <goocanvas.h>
#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "vectorstream.h"

// Operators from "vectorstream.h"
using BPN::operator<<;
using BPN::operator>>;

/**
 * Display
 *
 * +--------------------------------------------------------+
 * |               ^                                         |
 * |               |                                        |
 * |               3                                        |
 * |               |                            +--+        |
 * |<--2-->        v                            |  |        |
 * |       +--+--+--+                           +--+        |
 * |       |  |  |  |            +--+           |  |        |
 * |       +--+--+--+            |  |           +--+        |
 * |       |  |  |  |            +--+           |  |        |
 * |       +--+--+--+            |  |           +--+        |
 * |       |  |  |  |            +--+           |  |        |
 * |       +--+--+--+                           +--+        |
 * |       <1>                                  |  |        |
 * |                 <-----4---->               +--+        |
 * |                                                        |
 * +--------------------------------------------------------+
 */

// 1 size of the squares 
#define SQUARESIZE 20

// 2 horizontal grid offset
#define GRIDOFFSET_H 100

// 3 vertical grid offset
#define GRIDOFFSET_V 50

// 4 space between layers
#define LAYEROFFSET 100


static gboolean on_delete_event      (GtkWidget      *window,
                                      GdkEvent       *event,
                                      gpointer        unused_data);




class NeuralNetworkInterface;

struct InputNeuronInterfaceData
{
  InputNeuronInterfaceData( NeuralNetworkInterface* n, int i ) : nni(n), id(i)
  {}

  NeuralNetworkInterface* nni;
  int id;
};

struct MouseMode
{
  static guint currentMode;
};
guint MouseMode::currentMode = 0;

class NeuralNetworkInterface
{
public:
  NeuralNetworkInterface( BPN::Network* net, GtkWidget* canv ) : nn(net), canvas(canv)
    {
      int n = nn->getNumLayers();
      layers.resize( n, std::vector<GooCanvasItem*>() );
      buildInputGrid();
      buildLayersDisplay();

      // Reset button
      GooCanvasItem *root, *resetButton;
      root = goo_canvas_get_root_item (GOO_CANVAS (canvas));
      resetButton = goo_canvas_rect_new( root, 
                                        GRIDOFFSET_H + SQUARESIZE*(gridWidth/2-1), 
                                        GRIDOFFSET_V + SQUARESIZE*(gridHeight+2),
                                        2*SQUARESIZE, SQUARESIZE,
                                        "stroke-color", "black",
                                        "fill-color",   "red",
                                        NULL );
      g_signal_connect (resetButton, "button_press_event",
                        (GtkSignalFunc) resetInputNeurons, this );
      update();
    }

private:

  void buildInputGrid()
    {
      GooCanvasItem *root, *square;
      root = goo_canvas_get_root_item (GOO_CANVAS (canvas));
      int n = nn->getNumInputs();
      input.resize( n, 0.0 );
      int nRows = std::floor( std::sqrt(n) );
      int nCols = std::ceil(  std::sqrt(n) );
      int id = 0;
      for (int i=0; i<nRows; ++i)
        {
          for (int j=0; j<nCols; ++j)
            {
              int x = GRIDOFFSET_H + j*SQUARESIZE;
              int y = GRIDOFFSET_V + i*SQUARESIZE;
              square = goo_canvas_rect_new (root, x, y, SQUARESIZE, SQUARESIZE,
                                            //"line-width", 10.0,
                                            //"radius-x", 20.0,
                                            //"radius-y", 10.0,
                                            "stroke-color", "black",
                                            "fill-color", "white",
                                            NULL);
              InputNeuronInterfaceData* data = new InputNeuronInterfaceData(this, id);
              //g_signal_connect (square, "button_press_event",
              //                  (GtkSignalFunc) inputNeuronClicked, data );
              g_signal_connect (square, "button_press_event",
                                (GtkSignalFunc) inputNeuronPressed, data );
              //g_signal_connect (square, "button_release_event",
              //                  (GtkSignalFunc) inputNeuronRelease, data );
              gtk_widget_add_events((GtkWidget*)square, GDK_POINTER_MOTION_MASK);
              g_signal_connect (square, "motion_notify_event",
                                (GtkSignalFunc) inputNeuronTouched, data );
              layers[0].push_back( square );
              ++id;
            }
        }
      this->gridWidth = nRows;
      this->gridHeight = nCols;
    }

  void buildLayersDisplay()
    {
      GooCanvasItem *root, *square;
      root = goo_canvas_get_root_item (GOO_CANVAS (canvas));
      int n = nn->getNumLayers();
      const std::vector<int>& layersSizes = nn->getLayerSizes();
      for ( int i=1; i<n; ++i )
        {
          for ( int j=0; j<layersSizes[i]; ++j )
            {
              int x = GRIDOFFSET_H + SQUARESIZE*gridWidth + i*LAYEROFFSET + (i-1)*SQUARESIZE;
              int y = GRIDOFFSET_V + 2*j*SQUARESIZE;
              square = goo_canvas_rect_new ( root, x, y, SQUARESIZE, SQUARESIZE,
                                             //"line-width", 10.0,
                                             //"radius-x", 20.0,
                                             //"radius-y", 10.0,
                                             "stroke-color", "black",
                                             "fill-color", "white",
                                             NULL);
              layers[i].push_back( square );
            }
        }
    }

  void flipInput(unsigned int n)
    {
      if ( n < input.size() )
        {
          input[n] = 1.0 - input[n];
        }
    }

  static gboolean resetInputNeurons (GooCanvasItem  *view,
                                     GooCanvasItem  *target,
                                     GdkEventButton *event,
                                     gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      MouseMode::currentMode = 0;
      NeuralNetworkInterface* nni = (NeuralNetworkInterface*) user_data;
      nni->input.clear();
      nni->input.resize( nni->nn->getNumInputs(), 0.0 );
      nni->update();
    }

  static gboolean inputNeuronPressed (GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventButton *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) user_data;
      MouseMode::currentMode = (MouseMode::currentMode == event->button) ? 0 : event->button;
    }

  static gboolean inputNeuronClicked (GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventButton *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      std::cout << event->button << std::endl;
      InputNeuronInterfaceData * data = (InputNeuronInterfaceData*) user_data;
      if ( event->button == 1 )
        {
          data->nni->input[data->id] = 1.0;
        }
      else if ( event->button == 3 )
        {
          data->nni->input[data->id] = 0.0;
        }
      data->nni->update();
      std::cout << "Clicked on input neuron " << data->id << std::endl; 
    }

  static gboolean inputNeuronTouched (GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventMotion *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) event; (void) user_data;
      InputNeuronInterfaceData * data = (InputNeuronInterfaceData*) user_data;
      if ( MouseMode::currentMode == 1 )
        {
          data->nni->input[data->id] = 1.0;
          data->nni->update();
        }
      else if ( MouseMode::currentMode == 3 )
        {
          data->nni->input[data->id] = 0.0;
          data->nni->update();
        }
    }

  guint doubleToColor( double d )
    {
      d = ((d-neuronsMinValue) * ( neuronsMaxValue-neuronsMinValue )) + neuronsMinValue;
      int g = (int) (255*d);
      return (g<<24) + (g<<16) + (g<<8) + 255;
    }


  void update()
    {
      int n = nn->getNumLayers();
      const std::vector<int>& layersSizes = nn->getLayerSizes();
      nn->Evaluate( input );

      // input neurons
      for ( int j=0; j<layersSizes[0]; ++j )
        {
          double v = nn->getValue( 0, j );
          GooCanvasItem* square = layers[0][j];
          if ( v == 0.0 )
            {
              g_object_set(square, "fill-color", "white", NULL);
            }
          else
            {
              g_object_set(square, "fill-color", "black", NULL);
            }
        }
      for ( int i=1; i<n; ++i ) 
        {
          std::vector<double> thisLayer;
          for ( int j=0; j<layersSizes[i]; ++j )
            {
              double v = nn->getValue( i, j );
              neuronsMaxValue = std::max( neuronsMaxValue, v );
              neuronsMinValue = std::min( neuronsMinValue, v );
              GooCanvasItem* square = layers[i][j];
              g_object_set( square, "fill-color-rgba", doubleToColor(v), NULL );
              thisLayer.push_back(v);
            }
          //std::cout << "Layer " << i << " : " << thisLayer << std::endl;
        }
    }

  BPN::Network* nn;
  GtkWidget* canvas;
  std::vector< std::vector<GooCanvasItem*> > layers;
  std::vector<double> input;
  int gridWidth;
  int gridHeight;
  double neuronsMaxValue;
  double neuronsMinValue;
};
     

int
main (int argc, char *argv[])
{
  GtkWidget *window, *scrolled_win, *canvas;
  //GooCanvasItem *root;

  /* Initialize GTK+. */
  gtk_set_locale ();
  gtk_init (&argc, &argv);

  /* Create the window and widgets. */
  window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  gtk_window_set_default_size (GTK_WINDOW (window), 1600, 1000);
  gtk_widget_show (window);
  g_signal_connect (window, "delete_event", (GtkSignalFunc) on_delete_event,
                    NULL);

  scrolled_win = gtk_scrolled_window_new (NULL, NULL);
  gtk_scrolled_window_set_shadow_type (GTK_SCROLLED_WINDOW (scrolled_win),
                                       GTK_SHADOW_IN);
  gtk_widget_show (scrolled_win);
  gtk_container_add (GTK_CONTAINER (window), scrolled_win);

  canvas = goo_canvas_new ();
  gtk_widget_set_size_request (canvas, 1600, 1000);
  goo_canvas_set_bounds (GOO_CANVAS (canvas), 0, 0, 1600, 1000);
  gtk_widget_show (canvas);
  gtk_container_add (GTK_CONTAINER (scrolled_win), canvas);

  BPN::Network* nn;

  if ( argc == 1 )
    {
      nn = new BPN::Network( std::cin );
    }
  else 
    {
      std::fstream fs;
      fs.open( argv[1], std::fstream::in );
      nn = new BPN::Network( fs );
    }
  NeuralNetworkInterface nni( nn, canvas );

  /* Pass control to the GTK+ main event loop. */
  gtk_main ();

  delete nn;
  return 0;
}



/* This is our handler for the "delete-event" signal of the window, which
   is emitted when the 'x' close button is clicked. We just exit here. */
static gboolean
on_delete_event (GtkWidget *window,
                 GdkEvent  *event,
                 gpointer   unused_data)
{
  (void) window; (void) event; (void) unused_data;
  std::cout << "Quit" << std::endl;
  exit (0);
}

