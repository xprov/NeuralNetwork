/**
 * Fully Connected Neural Network
 */


ActivationFunctions = {
  "Sigmoid(1)" : function(x) {return 1.0 / (1.0 + Math.exp(-1.0 * x));}
}

class Neuron {
  constructor(fcnn, layer, index, isBiais) {
    this.fcnn = fcnn;
    this.layer = layer;
    this.index = index;
    this.isBiais = isBiais;
    this.value = (isBiais) ? 1.0 : 0.0;
    this.activation = 0.0;
    this.center = [0,0];
    this.topLeft = [0,0];
    this.bottomRight = [0,0];
  }

  update(activation, value) {
    this.activation = activation;
    this.value = value;
  }

  // todo fcnn should compute all there and simply affect the values to each neuron
  updatePosition() {
    var opt = this.fcnn.displayOptions;
    var size = opt.neuronsSize;
    var halfSize = Math.round(0.5*size);

    if (opt.inputDisposition === "column") {
      this.center = [
	Math.round((1.0+this.layer)*opt.horizontalSpacing),
	Math.round((1.0+this.index)*opt.verticalSpacing)
      ];
    }

    else if (opt.inputDisposition === "square") {
      var inputSquareSize = Math.ceil(Math.sqrt(fcnn.layers[0]));
      if (this.layer == 0) { // input node
	var row = Math.floor(this.index / inputSquareSize);
	var col = this.index % inputSquareSize;
	this.center = [
	  Math.round(opt.horizontalSpacing + col*size),
	  Math.round(opt.verticalSpacing + row*size)
	];
      } else {
	this.center = [
	  Math.round((1.0+this.layer)*opt.horizontalSpacing + (inputSquareSize-1)*size),
	  Math.round((1.0+this.index)*opt.verticalSpacing  )
	];
      }
    }
    this.topLeft     = [this.center[0]-halfSize, this.center[1]-halfSize];
    this.bottomRight = [this.center[0]+halfSize, this.center[1]+halfSize];
  }

  isClicked(x, y) {
    return (!this.isBiais) 
      && this.topLeft[0] <= x && x <= this.bottomRight[0] 
      && this.topLeft[1] <= y && y <= this.bottomRight[1];
  }

  toString() {
    return "Neuron(layer=" + this.layer 
      + ", index=" + this.index 
      + ", value=" + this.value 
      + ", center=" + this.center 
      + ")";
  }

  displaySelf(ctx) {
    var x = this.topLeft[0];
    var y = this.topLeft[1];
    var size = this.fcnn.displayOptions.neuronsSize;
    if (!this.isBiais) {
      ctx.beginPath();
      var intensity = Math.floor(255.0 * this.value);
      ctx.fillStyle = "rgb("+intensity+"," + intensity + "," + intensity + ")";
      ctx.fillRect(x, y, size, size);
      ctx.lineWidth = "2";
      ctx.strokeStyle = "black";
      ctx.strokeRect(x, y, size, size);
    } else {
      ctx.beginPath();
      ctx.fillStyle = "white";
      ctx.strokeStyle = "black";
      ctx.lineWidth = "2";
      ctx.arc(this.center[0], this.center[1], size/2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }


}

class FCNNDisplayOptions {
  constructor() {
    this.neuronsSize = 20;
    this.horizontalSpacing = 75;
    this.verticalSpacing = 50;
    this.inputDisposition = "column"; // "column" or "square"
    this.showConnexions = false;
    this.showBiais = false;
  }
}

class FCNN {
  constructor(text) {
    text = text.trim();
    var layersInText = null;
    this.activationFunctionsName = null;
    var weightsInText = null;
    text.split('\n').forEach( line => {
      if (line.startsWith("layerSizes")) {
        layersInText = line;
      } else if (line.startsWith("activation")) {
        this.activationFunctionsName = line.split(" ")[1];
      } else if (line.startsWith("weights")) {
        weightsInText = line;
      }
    });

    if (layersInText == null) {
      throw "Parsing error. Error reading layers' sizes";
    }
    if (this.activationFunctionsName == null) {
      throw "Parsing error. Error reading activation function";
    }
    if (weightsInText == null) {
      throw "Parsing error. Error reading weights";
    }

    // parse layers' sizes
    var p0 = layersInText.indexOf('[');
    var p1 = layersInText.indexOf(']');
    var layers = layersInText.substring(p0+1, p1);
    this.layers = []
    layers.split(',').forEach(k => {this.layers.push(parseInt(k))});
    this.numLayers = this.layers.length

    // parse activation function
    this.activationFunction = ActivationFunctions[this.activationFunctionsName];
    if (typeof this.activationFunction == 'undefined') {
      throw "Parsing error. Unknown activation function";
    }

    // built neurons
    // in this implementation, a neuron is just a number (float)
    // each layer of neurons is stored in an array
    this.neurons = [];
    for (var i=0; i<this.numLayers; i++) {
      var layer = [];
      for (var j=0; j<this.layers[i]; j++) {
	layer.push(new Neuron(this, i, j, false));
      }
      if (i != this.numLayers-1) {
	// to each layer, except the last one, we add an extra neuron with value 1.
	// This neuron represents the biais ans will never be updated.
	var biais = new Neuron(this, i, this.layers[i], true);
	layer.push(biais);
      }
      this.neurons.push(layer);
    }

    //this.layers.forEach(k => {this.neurons.push(new Array(k).fill(0.0));});
    //for (var i=0; i<this.neurons.length-1; i++) {
    //  this.neurons[i].push(1.0);
    //}


    // parse weights
    // this.weights[k][i][j] is the weight of the link from the i-th neuron of
    // the k-th level to the j-th neuron of the (k+1)-th layer.
    var weightsAsText = weightsInText.split(' ');
    var index = 1;
    this.weights = []
    for (var k=0; k<this.numLayers-1; k++) {
      var layersSize = this.layers[k];
      var nextLayersSize = this.layers[k+1];
      var outOfLayerK = [];
      for (var i=0; i<=layersSize; i++ ) {
        var outOfNeuronI = [];
        for (var j=0; j<nextLayersSize; j++) {
          if (index == weightsAsText.length) {
            throw "Insuficient number of weights given de layers' sizes";
          }
          var weight = parseFloat(weightsAsText[index++]);
          if (Number.isNaN(weight)) {
            throw "Invalid weight (=" + weightsAsText[index-1] + ") must be float";
          }
          outOfNeuronI.push(weight);
        }
        outOfLayerK.push(outOfNeuronI);
      }
      this.weights.push(outOfLayerK)
    }

    this.displayOptions = new FCNNDisplayOptions();
    this.updateNeuronsPositions();
  }

  updateNeuronsPositions() {
    this.neurons.forEach( layer => {layer.forEach( n => {n.updatePosition();})});
    //for (var i=0; i<this.numLayers; i++) {
    //  for (var j=0; j<this.layers[i]; j++) {
    //    this.neurons[i][j].updatePosition();
    //  }
    //}
  }

  /**
   * Assing the given values to the input neurons and updates all layers
   * consequently.
   */
  evaluate(input) {

    // update input neurons
    for (var i=0; i<this.layers[0]; i++) {
      if  (i < input.length)
	this.neurons[0][i].update(input[i], input[i]); // no actication function on input nodes
      else
	this.neurons[0][i].update(0.0, 0.0); // unspecified input, set to 0
    }

    // update all other layers
    for (var k=1; k<this.numLayers; k++) {
      for (var j=0; j<this.layers[k]; j++) {
	var activation = 0.0;
	for (var i=0; i<=this.layers[k-1]; i++) {
	  activation += this.neurons[k-1][i].value * this.weights[k-1][i][j];
	}
	this.neurons[k][j].update(activation, this.activationFunction(activation));
      }
    }

    return this.getOutput();
  }

  getOutput() {
    return this.neurons[this.numLayers-1];
  }


  toHTML() {
    return "<ul>"
    + "<li>  Nombre de couches : " + this.numLayers + "</li>"
    + "<li>  Taille de la couche d'entrée : " + this.layers[0] + "</li>"
    + "<li>  Taille de la couche de sortie : " + this.layers[this.numLayers-1] + "</li>"
    + "<li>  Taille de toutes les couches : (entrée) " + this.layers + " (sortie)</li>"
    + "<li>  Fonction d'activation : " + this.activationFunctionsName + "</li>"
    + "</ul>"
  }

  


  displaySelf() {
    myDisplayArea.clear();
    var ctx = myDisplayArea.context;
    
    // connexions between neurons
    if (this.displayOptions.showConnexions) {
      for (var k=1; k<this.numLayers; k++) {
	for (var j=0; j<this.layers[k]; j++) {
	  var to = this.neurons[k][j].center;
	  var numInPreviousLayer = (this.displayOptions.showBiais) ? this.layers[k-1]+1 : this.layers[k-1];
	  for (var i=0; i<numInPreviousLayer; i++) {
	    var from = this.neurons[k-1][i].center;
	    ctx.beginPath();
	    ctx.moveTo(from[0], from[1]);
	    ctx.lineTo(to[0], to[1]);
	    ctx.stroke(); 
	  }
	}
      }
    }

    // neurons
    this.neurons.forEach(layer => {layer.forEach(n => {
      if (this.displayOptions.showBiais || !n.isBiais) n.displaySelf(ctx)
    })});
  }
}




/**
 * Page interaction
 */

var fcnn = null;

/**
 * Load the neural network as described in the importBox text area.
 */
function importFCNN() {
  var inputText = document.getElementById("importBox").value;
  try {
    fcnn = new FCNN(inputText);
    fcnn.displaySelf();
    document.getElementById("fcnn-display-area").innerHTML = "Réseau de neuronnes importé avec succès :<br>" + fcnn.toHTML();
  } catch (err) {
    document.getElementById("fcnn-display-area").innerHTML = "Erreur à l'importation, réseau invalide. (" + err + ")";
  }
}

/**
 * Load the neural network as described in the importBox text area and
 * evaluates it using the input values specified in the inputValues.
 */
function evaluateFCNNwithTextInput() {
  if (fcnn == null) {
    document.getElementById("output-display-area").innerHTML = "Erreur, vous devez d'abord importer un RNCC valide.";
    return ;
  }
  var inputText = document.getElementById("importBox").value;
  var inputValues = document.getElementById("inputValues").value;
  var input = null;
  if (inputValues.indexOf(",") != -1) {
    input = inputValues.split(",").map(function(x) {return parseFloat(x)});
  } else {
    input = inputValues.split(" ").map(function(x) {return parseFloat(x)});
  }

  fcnn.evaluate(input);

  outputValues = fcnn.getOutput();
  outputHTML = outputValues.map(function(valeur, index) {
    return "Neuron " + (index+1) + " : " + valeur;
  }).join("<br>");
  document.getElementById("output-display-area").innerHTML = outputHTML;

  fcnn.displaySelf();
}

// Pressing the ENTER key in the import box will trigger the 'importer'
// button.
var inputValues = document.getElementById("importBox");
// Execute a function when the user releases a key on the keyboard
inputValues.addEventListener("keyup", function(event) {
  // Number 13 is the "Enter" key on the keyboard
  if (event.keyCode === 13) {
    // Cancel the default action, if needed
    event.preventDefault();
    // Trigger the button element with a click
    document.getElementById("importButton").click();
  }
}); 




// Pressing the ENTER key in the input box will trigger the 'Load and evaluate'
// button.
var inputValues = document.getElementById("inputValues");
// Execute a function when the user releases a key on the keyboard
inputValues.addEventListener("keyup", function(event) {
  // Number 13 is the "Enter" key on the keyboard
  if (event.keyCode === 13) {
    // Cancel the default action, if needed
    event.preventDefault();
    // Trigger the button element with a click
    document.getElementById("evaluateButton").click();
  }
}); 


var myDisplayArea = {
	canvas : document.createElement("canvas"),
	start : function() {
		this.canvas.width = Math.floor(window.innerWidth*0.9);
		this.canvas.height = Math.floor(this.canvas.width*0.5);
		this.context = this.canvas.getContext("2d");
		document.body.appendChild(this.canvas);
	},

	clear : function() {
	  var ctx = this.context;
	  ctx.fillStyle = "white";
	  ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
	}
};



// var c = document.getElementById("myCanvas");
// var ctx = c.getContext("2d");

myDisplayArea.start();
myDisplayArea.clear();

var hslider = document.getElementById("horizontalSpacingSlider");
hslider.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.horizontalSpacing = Number(this.value);
    fcnn.updateNeuronsPositions();
    fcnn.displaySelf();
  }
}

var vslider = document.getElementById("verticalSpacingSlider");
vslider.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.verticalSpacing = Number(this.value);
    fcnn.updateNeuronsPositions()
    fcnn.displaySelf();
  }
}

var sslider = document.getElementById("neuronsSizeSlider");
sslider.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.neuronsSize = Number(this.value);
    fcnn.updateNeuronsPositions();
    fcnn.displaySelf();
  }
}

var radioDispositionColonne = document.getElementById("radioDispositionColumn");
radioDispositionColonne.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.inputDisposition = this.value;
    fcnn.updateNeuronsPositions();
    fcnn.displaySelf();
  }
}

var radioDispositionSquare = document.getElementById("radioDispositionSquare");
radioDispositionSquare.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.inputDisposition = this.value;
    fcnn.updateNeuronsPositions();
    fcnn.displaySelf();
  }
}

var showConnexions = document.getElementById("showConnexions");
showConnexions.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.showConnexions = this.checked;
    fcnn.displaySelf();
  }
}

var showBiais = document.getElementById("showBiais");
showBiais.oninput = function() {
  if (fcnn != null) {
    fcnn.displayOptions.showBiais = this.checked;
    fcnn.displaySelf();
  }
}


function getCursorPosition(canvas, event) {
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    console.log("x: " + x + " y: " + y)
}



myDisplayArea.canvas.addEventListener('mousedown', function(e) {
    getCursorPosition(myDisplayArea.canvas, e)
})

myDisplayArea.canvas.addEventListener('mousemove', function(e) {
    getCursorPosition(myDisplayArea.canvas, e)
})
//// 
//// var mouseclick = false;
//// var downListener = function() {
////     mouseclick = true;
//// }
//// element.addEventListener('mousedown', downListener)
//// var moveListener = () => {
////     moved = true
//// }
//// element.addEventListener('mousemove', moveListener)
//// var upListener = () => {
////     if (moved) {
////         console.log('moved')
////     } else {
////         console.log('not moved')
////     }
//// }
//// element.addEventListener('mouseup', upListener)
//// 
//// // release memory
//// element.removeEventListener('mousedown', downListener)
//// element.removeEventListener('mousemove', moveListener)
//// element.removeEventListener('mouseup', upListener)
