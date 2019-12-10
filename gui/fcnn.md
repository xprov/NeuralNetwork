<!DOCTYPE html>
<head>
  <title> Réseaux de neuronnes </title>
  <meta charset="UTF-8">
  <link rel="stylesheet" href="fcnn.css">
</head>


Fun with Neural Networks
========================



Cette page est dédiée à l'évaluation de réseaux de neuronnes tels présentés 
par Guillaume Roy-Fortin lors de sa 
[célèbre série de séminaires](https://groyfortin.github.io/ml.html) qui furent
tenus à l'ÉTS en 2019.

Nous ne considérons ici que des _réseaux de neuronnes complètement connectés_
(en anglais, _fully connected neural networks_) que nous abrégeons par
__RNCC__.

Des exemples de réseaux déjà entraînés ainsi que des outils logiciels pour en
entrainer de nouveaux sont disponibles
[ici](https://github.com/xprov/NeuralNetwork).



 - Étape 1, importer un RNCC.

<html>
<div id="drop-area">
<form class="my-form">
<p>Déposer l'export du RNCC ici (drag and drop).</p>
<input type="file" id="fileElem" onchange="uploadFile(this.files)">
<label class="button" for="fileElem">Choisir le fichier</label>
</form>
</div>
<div id="copy-paste-area">
ou faire un copier/coller de l'export ici :<br>
<textarea id="importBox" rows="4" cols="100" wrap="off"> </textarea> 
<br>
<button id="importButton" onclick="importFCNN()">Importer</button> 
</div>
<div id="fcnn-display-area">
</div>

- Étape 2, saisir les valeurs des neuronnes en entrée : <input type="text" id="inputValues" >
<button id="evaluateButton" onclick="evaluateFCNN()">Évaluater</button> 
<br>
<p style="font-size:12px">
Entrer la liste des valeurs à affecter aux neuronnes d'entrée avec ou sans virgules. 
Par exemple : 0.1 0.2 0.3 ou 0.1,0.2,0.3
</p>
<div id="output-display-area"></div>

<p> test </p>

<div class="slidecontainer">
  <input type="range" min="1" max="150" value="75" class="slider" id="horizontalSpacingSlider">
  <input type="range" min="1" max="100" value="50" class="slider" id="verticalSpacingSlider">
  <input type="range" min="1" max="50" value="15" class="slider" id="neuronsSizeSlider">
</div>

<script src="fcnn.js"></script>
<script src="fileupload.js"></script>
</html>

