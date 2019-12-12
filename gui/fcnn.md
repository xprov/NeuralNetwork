<!DOCTYPE html>
<head>
  <title> Réseaux de neurones </title>
  <meta charset="UTF-8">
  <link rel="stylesheet" href="fcnn.css">
</head>


Fun with Neural Networks
========================



Cette page est dédiée à l'évaluation de réseaux de neurones tels présentés 
par Guillaume Roy-Fortin lors de sa 
célèbre [série de séminaires](https://groyfortin.github.io/ml.html) 
sur les mathématiques des réseaux de neurones.

Nous ne considérons ici que des _réseaux de neurones complètement connectés_
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

- Étape 2, saisir les valeurs des neurones en entrée : <input type="text" id="inputValues" >
<button id="evaluateButton" onclick="evaluateFCNNwithTextInput()">Évaluer</button> 

	<font size="1"> 
	Entrer la liste des valeurs à affecter aux neurones d'entrée avec ou sans virgules. 
	Par exemple : 0.1 0.2 0.3 ou 0.1,0.2,0.3
	</font> 

<div id="output-display-area"></div>


<div class="slidecontainer">
<table>
  <col align="left">
  <col align="left">
  <col align="left">
  <col align="left">
  <col align="left">
  <col align="left">
  <col align="left">
  <tr>
    <td>Espacement horizontal </td>
    <td><input type="range" min="1" max="150" value="75" class="slider" id="horizontalSpacingSlider"></td>
    <td></td>
    <td><input type="checkbox" id="showConnexions" name="showConnexions" value="showConnexions"></td>
    <td>Afficher les liens</td>
    <td></td>
    <td><button id="setToZero" onclick="setAllInputsToZero()">Tout mettre à zéro</button> 
  </tr>
  <tr>
    <td>Espacement vertical</td>
    <td><input type="range" min="1" max="100" value="50" class="slider" id="verticalSpacingSlider"></td>
    <td></td>
    <td><input type="checkbox" id="showBiais" name="showBiais" value="showBiais"></td>
    <td>Afficher les biais</td>
    <td></td>
    <td><button id="setToOne" onclick="setAllInputsToOne()">Tout mettre à un</button> 
  </tr>
  <tr>
    <td>Taille des neurones</td>
    <td><input type="range" min="1" max="50" value="15" class="slider" id="neuronsSizeSlider"></td>
  </tr>
<tr> 
  <td>Disposition des neurones d'entrée</td>
  <td> 
    <input type="radio" id="radioDispositionColumn" name="radioDispo" value="column" checked="checked"> Colonne
    <input type="radio" id="radioDispositionSquare" name="radioDispo" value="square"> Carré
  </td>
</tr>
</table>
</div>

<script src="fcnn.js"></script>
<script src="fileupload.js"></script>
</html>

