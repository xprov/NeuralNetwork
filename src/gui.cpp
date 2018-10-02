//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include <stdlib.h>
#include <goocanvas.h>
#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "vectorstream.h"
#include "cmdParser.h"

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
  NeuralNetworkInterface( BPN::Network* net, 
                          GtkWidget* canv, 
                          const std::vector<std::string>& outputLabels )
    : nn(net) , canvas(canv)
    {
      int n = nn->getNumLayers();
      layers.resize( n, std::vector<GooCanvasItem*>() );
      buildInputGrid();
      buildLayersDisplay( outputLabels );

      GooCanvasItem *root, *resetButton, *noisifyButton, *resetText, *noisifyText,
                    *mouseActionText, *translateUp, *translateDown, *translateLeft, 
                    *translateRight;
      int x, y, width, height;

      // get the root canvas item where everithing else is added
      root = goo_canvas_get_root_item (GOO_CANVAS (canvas));

      // Reset button
      x = GRIDOFFSET_H + SQUARESIZE*(gridWidth/2-2); 
      y = GRIDOFFSET_V + SQUARESIZE*(gridHeight+2);
      width  = 4*SQUARESIZE;
      height = 2*SQUARESIZE;
      resetButton = goo_canvas_rect_new( root, x, y, width, height, 
                                         "stroke-color", "black",
                                         "fill-color",   "red",
                                         NULL );
      g_signal_connect (resetButton, "button_press_event",
                        (GtkSignalFunc) resetInputNeurons, this );

      resetText = goo_canvas_text_new( root,
                                       "Reset",
                                        x + width/2, y + height/2,
                                        -1, GTK_ANCHOR_CENTER, NULL);

      g_signal_connect (resetText, "button_press_event",
                        (GtkSignalFunc) resetInputNeurons, this );

      // Noisify button
      x = GRIDOFFSET_H + SQUARESIZE*(gridWidth/2-8); 
      y = GRIDOFFSET_V + SQUARESIZE*(gridHeight+2);
      width  = 4*SQUARESIZE;
      height = 2*SQUARESIZE;
      noisifyButton = goo_canvas_rect_new( root, x, y, width, height, 
                                         "stroke-color", "black",
                                         "fill-color",   "red",
                                         NULL );
      g_signal_connect (noisifyButton, "button_press_event",
                        (GtkSignalFunc) noisifyInputNeurons, this );

      noisifyText = goo_canvas_text_new( root,
                                       "Noisify",
                                        x + width/2, y + height/2,
                                        -1, GTK_ANCHOR_CENTER, NULL);

      g_signal_connect (noisifyText, "button_press_event",
                        (GtkSignalFunc) noisifyInputNeurons, this );

      int translationButtonsX = GRIDOFFSET_H + SQUARESIZE*(gridWidth/2+6); 
      int translationButtonsY = GRIDOFFSET_V + SQUARESIZE*(gridHeight+3); 

      // Translation buttons
      // UP
      x = translationButtonsX;
      y = translationButtonsY - height; 
      width  = 2*SQUARESIZE;
      translateUp = goo_canvas_rect_new( root, x, y, width, height, 
                                         "stroke-color", "black",
                                         "fill-color",   "white",
                                         NULL );
      g_signal_connect (resetButton, "button_press_event",
                        (GtkSignalFunc) translateInputUp, this );

      resetText = goo_canvas_text_new( root,
                                       "↑",
                                        x + width/2, y + height/2,
                                        -1, GTK_ANCHOR_CENTER, NULL);

      g_signal_connect (resetText, "button_press_event",
                        (GtkSignalFunc) translateInputUp, this );

      // Down
      x = translationButtonsX;
      y = translationButtonsY + height; 
      translateDown = goo_canvas_rect_new( root, x, y, width, height, 
                                         "stroke-color", "black",
                                         "fill-color",   "white",
                                         NULL );
      g_signal_connect (resetButton, "button_press_event",
                        (GtkSignalFunc) translateInputUp, this );

      resetText = goo_canvas_text_new( root,
                                       "↓",
                                        x + width/2, y + height/2,
                                        -1, GTK_ANCHOR_CENTER, NULL);

      g_signal_connect (resetText, "button_press_event",
                        (GtkSignalFunc) translateInputDown, this );

      // Left
      x = translationButtonsX - width;
      y = translationButtonsY;
      translateLeft = goo_canvas_rect_new( root, x, y, width, height, 
                                         "stroke-color", "black",
                                         "fill-color",   "white",
                                         NULL );
      g_signal_connect (resetButton, "button_press_event",
                        (GtkSignalFunc) translateInputUp, this );

      resetText = goo_canvas_text_new( root,
                                       "←",
                                        x + width/2, y + height/2,
                                        -1, GTK_ANCHOR_CENTER, NULL);

      g_signal_connect (resetText, "button_press_event",
                        (GtkSignalFunc) translateInputLeft, this );

      // Right
      x = translationButtonsX + width;
      y = translationButtonsY;
      translateRight = goo_canvas_rect_new( root, x, y, width, height, 
                                         "stroke-color", "black",
                                         "fill-color",   "white",
                                         NULL );
      g_signal_connect (resetButton, "button_press_event",
                        (GtkSignalFunc) translateInputUp, this );

      resetText = goo_canvas_text_new( root,
                                       "→",
                                        x + width/2, y + height/2,
                                        -1, GTK_ANCHOR_CENTER, NULL);

      g_signal_connect (resetText, "button_press_event",
                        (GtkSignalFunc) translateInputRight, this );

      // Display the action that is currently performed when the mouse pointer
      // touches an input neuron
      x = GRIDOFFSET_H + SQUARESIZE*(gridWidth/2-2) + width/2; 
      y = GRIDOFFSET_V + SQUARESIZE*(gridHeight+2) + 2*SQUARESIZE + height/2;
      mouseActionText = goo_canvas_text_new( root,
                                             "Action",
                                             x - SQUARESIZE/2, y + SQUARESIZE/2,
                                             -1, GTK_ANCHOR_EAST, NULL );

      mouseActionSquare = goo_canvas_rect_new( root, x, y,
                                              SQUARESIZE, SQUARESIZE,
                                              "stroke-color", "white",
                                              "fill-color", "white",
                                              NULL );

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

  void buildLayersDisplay( const std::vector<std::string>& outputLabels)
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
              if ( i == n-1 )
                {
                  try {
                      const char* text = outputLabels.at(j).c_str();
                      GooCanvasItem * label = goo_canvas_text_new( root, text, x + 2*SQUARESIZE, y+SQUARESIZE/2,
                                                                  -1, GTK_ANCHOR_WEST, NULL );
                  }
                  catch ( std::out_of_range e )
                    {
                    }
                }
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

  static gboolean translateInputUp( GooCanvasItem  *view,
                                    GooCanvasItem  *target,
                                    GdkEventButton *event,
                                    gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      NeuralNetworkInterface* nni = (NeuralNetworkInterface*) user_data;
      int nRows = nni->gridHeight;
      int nCols = nni->gridWidth;
      for ( int i=0; i<nRows-1; ++i )
        {
          for ( int j=0; j<nCols; ++j )
            {
              nni->input.at( i*nCols + j ) = nni->input.at( (i+1)*nCols + j );
            }
        }
      for ( int j=0; j<nCols; ++j )
        {
              nni->input.at( (nRows-1)*nCols + j ) = 0;
        }
      nni->update();
    }

  static gboolean translateInputDown( GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventButton *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      NeuralNetworkInterface* nni = (NeuralNetworkInterface*) user_data;
      int nRows = nni->gridHeight;
      int nCols = nni->gridWidth;
      for ( int i=nRows-1; i>=1; --i )
        {
          for ( int j=0; j<nCols; ++j )
            {
              nni->input.at( i*nCols + j ) = nni->input.at( (i-1)*nCols + j );
            }
        }
      for ( int j=0; j<nCols; ++j )
        {
              nni->input.at( j ) = 0;
        }
      nni->update();
    }

  static gboolean translateInputLeft( GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventButton *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      NeuralNetworkInterface* nni = (NeuralNetworkInterface*) user_data;
      int nRows = nni->gridHeight;
      int nCols = nni->gridWidth;
      for ( int j=0; j<nCols-1; ++j )
        {
          for ( int i=0; i<nRows; ++i )
            {
              nni->input.at( i*nCols + j ) = nni->input.at( i*nCols + j+1 );
            }
        }
      for ( int i=0; i<nRows; ++i )
        {
              nni->input.at( i*nCols + nCols-1 ) = 0;
        }
      nni->update();
    }

  static gboolean translateInputRight( GooCanvasItem  *view,
                                       GooCanvasItem  *target,
                                       GdkEventButton *event,
                                       gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      NeuralNetworkInterface* nni = (NeuralNetworkInterface*) user_data;
      int nRows = nni->gridHeight;
      int nCols = nni->gridWidth;
      for ( int j=nCols-1; j>=1; --j )
        {
          for ( int i=0; i<nRows; ++i )
            {
              nni->input.at( i*nCols + j ) = nni->input.at( i*nCols + j-1 );
            }
        }
      for ( int i=0; i<nRows; ++i )
        {
              nni->input.at( i*nCols ) = 0;
        }
      nni->update();
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

  static gboolean noisifyInputNeurons (GooCanvasItem  *view,
                                     GooCanvasItem  *target,
                                     GdkEventButton *event,
                                     gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      MouseMode::currentMode = 0;
      NeuralNetworkInterface* nni = (NeuralNetworkInterface*) user_data;
      for (int i=0; i<5; ++i) 
        {
          int index = rand() % nni->input.size();
          nni->input[index] = 1 - nni->input[index];
        }
      nni->update();
    }

  static gboolean inputNeuronPressed (GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventButton *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) user_data;
      InputNeuronInterfaceData * data = (InputNeuronInterfaceData*) user_data;
      if ( MouseMode::currentMode != 1 && event->button == 1 )
        {
          data->nni->input[data->id] = 1.0;
          MouseMode::currentMode = 1;
        }
      else if ( MouseMode::currentMode != 3 && event->button == 3 )
        {
          data->nni->input[data->id] = 0.0;
          MouseMode::currentMode = 3;
        }
      else if ( MouseMode::currentMode == event->button )
        {
          MouseMode::currentMode = 0;
        }
      data->nni->update();
    }

  static gboolean inputNeuronClicked (GooCanvasItem  *view,
                                      GooCanvasItem  *target,
                                      GdkEventButton *event,
                                      gpointer        user_data)
    {
      (void) view; (void) target; (void) event;
      //std::cout << event->button << std::endl;
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
      //std::cout << "MouseMode = " << MouseMode::currentMode << std::endl;
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

      // Hidden and output layers
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

      // Display mouse action button
      if ( MouseMode::currentMode == 0 ) 
        {
          g_object_set( mouseActionSquare, "fill-color", "grey", NULL );
          g_object_set( mouseActionSquare, "stroke-color", "grey", NULL );
        }
      else if ( MouseMode::currentMode == 1 )
        {
          g_object_set( mouseActionSquare, "fill-color", "black", NULL );
          g_object_set( mouseActionSquare, "stroke-color", "black", NULL );
        }
      else if ( MouseMode::currentMode == 3 )
        {
          g_object_set( mouseActionSquare, "fill-color", "white", NULL );
          g_object_set( mouseActionSquare, "stroke-color", "black", NULL );
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
  GooCanvasItem* mouseActionSquare;
};
     

int main (int argc, char *argv[])
{
  cli::Parser cmdParser( argc, argv );
  cmdParser.set_required<std::string>( "nn", "NeuralNetwork", "Path to the neural network export file. ( \"-\" stands for stdin )" );
  cmdParser.set_optional<std::string>( "l", "outputLabels", "[]", "Comma separated list of the labels of the output neurons" );
  cmdParser.set_optional<std::string>( "s", "size", "1600,1000", "Dimensions of the window, width and height separated by a comma (no blanks).");

  if ( !cmdParser.run() )
    {
      std::cout << "Invalid command line arguments";
      return 1;
    }

  const std::string pathToNN         = cmdParser.get<std::string>( "nn" );
  const std::string labelsAsText     = cmdParser.get<std::string>( "l" );
  const std::string windowSizeAsText = cmdParser.get<std::string>( "s" );


  std::vector<std::string> labels;
  std::stringstream labelsStream( labelsAsText );
  labelsStream >> labels;

  std::vector<int> windowSize;
  std::stringstream windowSizesStream( windowSizeAsText );
  windowSizesStream >> windowSize;
  int width  = windowSize[0];
  int height = windowSize[1];

  std::cout << "path : " << pathToNN << std::endl;
  std::cout << "labels : " << labels << std::endl;
  std::cout << "size : " << width << " x " << height  << std::endl;


  BPN::Network* nn;
  if ( pathToNN.compare( "-" ) == 0 )
    {
      nn = new BPN::Network( std::cin );
    }
  else 
    {
      std::fstream fs;
      fs.open( pathToNN, std::fstream::in );
      nn = new BPN::Network( fs );
    }

  GtkWidget *window, *scrolled_win, *canvas;
  //GooCanvasItem *root;

  /* Initialize GTK+. */
  gtk_set_locale ();
  gtk_init (&argc, &argv);

  /* Create the window and widgets. */
  window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  gtk_window_set_default_size (GTK_WINDOW (window), width, height);
  gtk_widget_show (window);
  g_signal_connect (window, "delete_event", (GtkSignalFunc) on_delete_event,
                    NULL);

  scrolled_win = gtk_scrolled_window_new (NULL, NULL);
  gtk_scrolled_window_set_shadow_type (GTK_SCROLLED_WINDOW (scrolled_win),
                                       GTK_SHADOW_IN);
  gtk_widget_show (scrolled_win);
  gtk_container_add (GTK_CONTAINER (window), scrolled_win);

  canvas = goo_canvas_new ();
  gtk_widget_set_size_request (canvas, width, height );
  goo_canvas_set_bounds (GOO_CANVAS (canvas), 0, 0, width, height );
  gtk_widget_show (canvas);
  gtk_container_add (GTK_CONTAINER (scrolled_win), canvas);

  NeuralNetworkInterface nni( nn, canvas, labels );

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

