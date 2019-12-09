// ************************ Drag and drop ***************** //
let dropArea = document.getElementById("drop-area")

// Prevent default drag behaviors
;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)   
  document.body.addEventListener(eventName, preventDefaults, false)
})

// Highlight drop area when item is dragged over it
;['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false)
})

;['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false)
})

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false)

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}

function highlight(e) {
  dropArea.classList.add('highlight')
}

function unhighlight(e) {
  dropArea.classList.remove('active')
}

function handleDrop(e) {
  var dt = e.dataTransfer
  var files = dt.files

  handleFiles(files)
}

let uploadProgress = []
let progressBar = document.getElementById('progress-bar')

function initializeProgress(numFiles) {
  progressBar.value = 0
  uploadProgress = []

  for(let i = numFiles; i > 0; i--) {
    uploadProgress.push(0)
  }
}

function updateProgress(fileNumber, percent) {
  uploadProgress[fileNumber] = percent
  let total = uploadProgress.reduce((tot, curr) => tot + curr, 0) / uploadProgress.length
  console.debug('update', fileNumber, percent, total)
  progressBar.value = total
}

function handleFiles(files) {
  files = [...files]
  initializeProgress(files.length)
  files.forEach(uploadFile)
  files.forEach(previewFile)
}

function previewFile(file) {
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function() {
    let img = document.createElement('img')
    img.src = reader.result
    document.getElementById('gallery').appendChild(img)
  }
}

function uploadFile(file, i) {
  var url = 'https://api.cloudinary.com/v1_1/joezimim007/image/upload'
  var xhr = new XMLHttpRequest()
  var formData = new FormData()
  xhr.open('POST', url, true)
  xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest')

  // Update progress (can be used to show progress indicator)
  xhr.upload.addEventListener("progress", function(e) {
    updateProgress(i, (e.loaded * 100.0 / e.total) || 100)
  })

  xhr.addEventListener('readystatechange', function(e) {
    if (xhr.readyState == 4 && xhr.status == 200) {
      updateProgress(i, 100) // <- Add this
    }
    else if (xhr.readyState == 4 && xhr.status != 200) {
      // Error. Inform the user
    }
  })

  formData.append('upload_preset', 'ujpu6gyk')
  formData.append('file', file)
  xhr.send(formData)
}


/**
 * Fully Connected Neural Network
 */


ActivationFunctions = {
  "Sigmoid(1)" : function(x) {return 1.0 / (1.0 + Math.exp(-1.0 * x));}
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
    this.neurons = []
    this.layers.forEach(k => {this.neurons.push(new Array(k).fill(0.0));});
    // to each layer, except the last one, we add an extra neuron with value 1.
    // This neuron represents the biais ans will never be updated.
    for (var i=0; i<this.neurons.length-1; i++) {
      this.neurons[i].push(1.0);
    }


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
  }

  /**
   * Assing the given values to the input neurons and updates all layers
   * consequently.
   */
  evaluate(input) {
    // update input neurons
    for (var i=0; i<this.layers[0]; i++) {
      this.neurons[0][i] = input[i];
    }
    // update all other layers
    for (var k=1; k<this.numLayers; k++) {
      for (var j=0; j<this.layers[k]; j++) {
        var activation = 0.0;
        for (var i=0; i<=this.layers[k-1]; i++) {
          //console.log("Layer=" + k
          //  + ", actualIdx=" + j
          //  + ", prevIdx=" + i
          //  + ", activation=" + this.neurons[k-1][i] + " * "  + this.weights[k-1][i][j]);
          activation += this.neurons[k-1][i] * this.weights[k-1][i][j];
        }
        this.neurons[k][j] = this.activationFunction(activation);
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
    document.getElementById("fcnn-display-area").innerHTML = "Réseau de neuronnes importé avec succès :<br>" + fcnn.toHTML();
  } catch (err) {
    document.getElementById("fcnn-display-area").innerHTML = "Erreur à l'importation, réseau invalide. (" + err + ")";
  }
  console.log(fcnn);
}

/**
 * Load the neural network as described in the importBox text area and
 * evaluates it using the input values specified in the inputValues.
 */
function evaluateFCNN() {
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
    document.getElementById("importFCNN").click();
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

