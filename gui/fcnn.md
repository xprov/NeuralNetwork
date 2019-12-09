<!DOCTYPE html>
<head>
  <title> Réseaux de neuronnes </title>
  <meta charset="UTF-8">
  <link rel="stylesheet" href="tuto.css">
</head>


Fun with Neural Networks
========================



Cette page est dédiée à l'évaluation de réseaux de neuronnes tels présentés 
par Guillaume Roy-Fortin lors de sa 
[célèbre série de séminaires](https://groyfortin.github.io/ml.html) présentés
en 2019.

Nous ne considérons ici que des _réseaux de neuronnes complètement connectés_
(en anglais, _fully connected neural networks_) que nous abrégeons par
__RNCC__.

Des exemples de tels réseaux déjà entraînés ainsi que des outils logiciels pour en entrainer de nouveaux sont disponibles [ici](https://github.com/xprov/NeuralNetwork).

 - Étape 1, importer un RNCC.

  <div id="drop-area">
  <form class="my-form">
  <p>Déposer l'export du RNCC ici (drag and drop).</p>
  <input type="file" id="fileElem" multiple accept="image/*" onchange="handleFiles(this.files)">
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
   <div id="output-display-area"></p>





<script src="tuto.js"></script>

