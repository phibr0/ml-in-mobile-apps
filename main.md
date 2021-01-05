---
author:
- ASGSG Informatik, 2020/2021
bibliography:
- references.bib
date: Phillip Bronzel
title: Machine Learning in Smartphone Apps
---

![image](titlepage.png) [@titlepageimage]

Hiermit versichere ich, dass ich die Arbeit selbstständig verfasst, dass
ich keine anderen Quellen und Hilfsmittel als die angegebenen benutzt
und die Stellen der Arbeit, die anderen Quellen dem Wortlaut oder Sinn
nach entnommen sind, in jedem einzelnen Fall unter Angabe von Quellen
kenntlich gemacht habe.

# Einführung

Im Enterprise und Forschungsbereich spielt Machine Learning schon seit
vielen Jahren eine bedeutende Rolle. Doch wie kann es dem Endnutzer, zum
Beispiel in mobilen Apps, weiterhelfen?

## Wahl des Themas

Seitdem ich im Jahr 2017 meinen ersten richtigen Kontakt mit der
Programmierung von Microcontrollern (Arduinos) hatte, habe ich mich
stark für die Entwicklung von Software interessiert. Dies war ein guter
Einstieg, da man dort schnell und recht einfach Ergebnisse, wie zum
Beispiel eine blinkende LED, erzielt.

Auch habe ich mich seitdem immer für die "neuen großen Technologien" wie
Blockchain oder Machine Learning interessiert. Zum Thema Machine
Learning habe ich zuvor noch nicht viel gemacht, daher ergriff ich die
Chance dieses Jahr meine Facharbeit über dieses Thema zu schreiben.

> AI is profound, and we are at a point---and it will get better and
> better over time---where the GPU is getting so powerful there's so
> much capability to do unbelievable things. What all of us have to do
> is to make sure we are using AI in a way that is for the benefit of
> humanity, not to the detriment of humanity.[^1]

Ich persönlich finde dieses Zitat sehr wichtig; es ist jetzt über 3
Jahre alt und bis heute hat sich enorm viel in diesem Bereich getan. Wir
haben nun GPU's, welche speziell auf mathematische Berechnungen mit
Tensoren optimiert sind und so das Trainieren von Neuronalen Netzen um
ein Vielfaches beschleunigen.[^2]

Des weiteren ist es mir, genauso wie Cook, wichtig, diese mächtige
Technologie nicht zu missbrauchen[^3], sondern gute Dinge mit ihr zu
schaffen: wie beispielsweise im Bereich der Medizin. In diesem Bereich
wurden schon viele beachtliche Anwendungszwecke gefunden, so hat
Google's Tochterfirma DeepMind im Dezember 2020 eine Technologie[^4]
präsentiert, welche das Falten von Proteinen akkurat prognostizieren
kann; dies war vorher nur sehr langsam und deutlich ungenauer
möglich.[^5]

## Ziel der Arbeit

Mein persönliches Ziel ist es, mehr über den Aufbau von Neuronalen
Netzen und die Funktionsweise von Machine Learning zu lernen. Außerdem
möchte ich auch ein praktisches Ergebniss haben, dafür habe ich im
Kapitel !TODO! eine App entwickelt, welche dem Nutzer mehr Informationen
über Produkte beim einkaufen liefern soll.

# Neuronale Netzwerke

## Wichtigste Ereignisse in der Geschichte

Im Jahr 1943 wurde die erste Arbeit darüber geschrieben, wie Neuronen im
Gehirn funktionieren könnten und die Autoren Warren McCulloch und Walter
Pitts experimentierten sogar damit diese mit elektronischen
Schaltkreisen nachzubauen.[^6]

In den 1950er Jahren haben Forscher von IBM daran gearbeitet ein
Neuronales Netzwerk mit einem Computer zu simulieren. Der Versuch
scheiterte allerdings.[^7]

Immer wieder gab es kleinere Forschungsprojekte, ein sehr großer
Durchbruch war aber 1975 die Entwicklung eines "Backpropagation"
Algorithmus durch den Wissenschaftler Paul Werbos. Ähnliche Algorithmen
wurden wiederholt und unabhängig entwickelt, aber Werbos' Algorithmus
war der erste mit großer Bedeutung.[^8] Das Prinzip des Algorithmus wird
auch heute noch verwendet, es ist dieser Algorithmus der dem Neuronalen
Netzwerk das selbstständige Lernen ermöglicht.[^9]

In 1998 veröffentlichte Yann LeCun und sein Team eine Arbeit über die
Anwendung eines "Convolutional Neural Networks[^10]" zur Erkennung von
geschriebenen Zeichen in einem Dokument.[^11] Diese Arbeit gilt als
Ursprung des, für beispielsweise Bilderkennungs Software gut geeignete,
CNNs und Weiterentwicklungen werden auch heute noch verwendet.

Obwohl ein großes Potenzial erkannt wurde, war es über die nächsten
Jahre wieder recht still. Der nächste große Durchbruch passierte in 2012
als Geoffrey Hinton ein Modell entwickelte, was die Fehlerquote in einer
öffentlichen Challenge für Bilderkennung beinahe halbierte.[^12] Der
Grund dafür waren mehrere fundamentale Neuerungen aus dem Bereich Deep
Learning; die wahrscheinlich größte Änderung: Starke Parallelisierung
des Backpropagation-Prozesses, durch Verschiebung der Last von der CPU
auf die GPU. Aufgrund der starken Überlegenheit eines Grafikprozessors
in parallelisierten Prozessen, wie die benötigten Tensormultiplikationen
durch die deutlich größere Anzahl an (dafür schwächeren) Kernen im
Vergleich zu einer herkömmlichen CPU, kann ein Neuronales Netzwerk
mehrere hundertmal schneller trainiert werden.

Heute gibt es (vergleichsweise) simple Frameworks, wie das im Jahr 2015
erschienende TensorFlow oder PyTorch aus 2016, welche das erstellen,
trainieren und verwenden von Neuronales Netzwerk enorm vereinfachen. Ihr
Funktionsumfang wächst durch die große Open-Source Community ständig.

19402020

## Aufbau

r87mm

In Abbildung 2 sieht man den Aufbau eines herkömmlichen künstlichen
Neuronalen Netzwerks, so wie es noch vor 40 Jahren verwendet wurde. In
der Grafik erkennt man drei Layer mit einer x-beliebigen Anzahl
Neuronen, welche untereinander mit jeweils allen Neuronen der vorigen
und nächsten Layer verbunden sind. Im Gegensatz zu einem biologischen
Neuron, welches nur aktiv oder inaktiv sein kann, kann ein künstliches
Neuron einen Zustand in Form eines Wertes von ${0 \leq x \leq 1}$ haben.
Jede Verbindung hat einen Weight Paramter und auch jedes Neuron hat
einen Bias. Die Anzahl der Hidden Layer kann an das Ziel angepasst und
ausgewählt werden und auch die Anzahl der einzelnen Neuronen ist erstmal
beliebig, als Faustregel für gute Ergebnisse gilt aber:

-   Die Anzahl der Neuronen in dem Hidden Layer sollte zwischen der
    Größe des Input und Output Layers liegen.

-   Die Anzahl der Neuronen in dem Hidden Layer sollte etwa
    $\frac{2}{3}$ der Größe des Input Layers plus der Größe des Output
    Layers entsprechen.

-   Die Anzahl der Neuronen in einem Hidden Layer sollte weniger als die
    Hälfte der Größe des Input Layers sein.[^13]

### Erstellung eines Neuronalen Netzwerks anhand eines Beispiels

Als Beispiel für ein Neuronales Netzwerk, welches darauf ausgelegt ist,
geschriebene Ziffern aus Bildern mit 24x24 Pixeln und nur Graustufen zu
erkennen wäre dann: Ein Input Layer mit $24^2$ Neuronen, jeweils für
jeden Pixel, welche jeweils eine Aktivierung zwischen 0 (komplett weiß)
und 1 (komplett schwarz) haben können, eines. Genau 10 Neuronen im
Output Layer, für jedes Zahlzeichen eines. Schließlich muss die Anzahl
der Hidden Layer und Neuronen festgelegt werden. Ich wähle als Beispiel
2 Layer mit jeweils 16 Neuronen, die Neuronen-Anzahl kann aber auch
unterschiedlich sein. Auch die Weights und Biases werden zunächst
zufällig ausgewählt, die Werte werden dann später im
Trainingsprozess[^14] angepasst. Die Eingabe in das Netzwerk ist also
ein Zweidimensionaler Tensor, oder auch eine Matrix, mit den Bilddaten
und die Ausgabe des Netzwerks ist ein eindimensionaler Tensor, oder auch
ein Vektor, mit den Prognosen.

$$\begin{bmatrix}
        i_{0,0} & \ldots & i_{0,23} \\
        \vdots  & \ddots & \vdots  \\
        i_{23,0} & \ldots & i_{23,23}
    \end{bmatrix}
    \Longrightarrow
    \text{Hidden Layer}
    \Longrightarrow
    \begin{bmatrix}
        o_0 \\
        \vdots \\
        o_9
    \end{bmatrix}$$

## Funktionsweise

r60mm

Ein Neuronales Netzwerk kann man sich eigentlich als eine große
Mathematische Funktion vorstellen. In dem zuvor genannten Beispiel wäre
es eine Funktion mit 576 Variablen und 10 Ergebnissen. Gibt man dieser
Funktion nun ein Bild, beziehungsweise 576 Werte als Input, so werden
von links nach rechts alle Weights $w$ und Biases $b$ zusammen mit dem
vorigen Aktivierungswerten $a$ berechnet. Da ein Neuron aber nur Werte
im Bereich $0\leq x \leq 1$ haben kann[^15], so wird das Ergebniss noch
mithilfe einer Aktivierungsfunktion in diesen Bereich umgewandelt.[^16]
Eine früher Häufig verwendete Funktion ist dabei die Sigmoidfunktion,
siehe Abbildung [\[sigmoid\]](#sigmoid){reference-type="ref"
reference="sigmoid"}.[^17] Es gibt aber auch noch eine Vielzahl weiterer
Funktionen, wie die heute häufig verwendete ReLU Funktion[^18], welche
den Trainingprozess durch die einfachere Funktion beschleunigt.[^19] Die
daraus resultierende Funktion würde in etwa so aussehen:[^20]

$$\label{funktion1}
    \sigma(w_1a_1+w_2a_2+w_3a_3+ \ldots +w_na_n+b)$$

Um mit dieser Formel alle Aktivierungen auf einmal berechnen zu können
verwendet man folgende Funktion, in welcher alle Weights und Biases in
Spalten-Vektoren zusammengefasst werden. Die Hochzeichen sind keine
Exponenten sondern gelten als Bezeichnung für den Layer, hier
beispielsweise 0 und 1. Das Ergebniss dieser Funktion ist ein Vektor mit
allen Aktivierungen des darauf folgenden Layers.

$$\label{funktion2}
    \sigma
    \begin{pmatrix}
        \begin{bmatrix}
            w_{0,0} & w_{0,1} & \ldots & w_{0,n} \\
            w_{1,0} & w_{1,1} & \ldots & w_{1,n} \\
            \vdots  & \vdots  & \ddots & \vdots  \\
            w_{k,0} & w_{k,1} & \ldots & w_{k,n}
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            a_0^{(0)} \\a_1^{(0)}\\\vdots\\a_n^{(0)}
        \end{bmatrix}
        +
        \begin{bmatrix}
            b_0 \\b_1\\\vdots\\b_k
        \end{bmatrix}
    \end{pmatrix}
    =
    a^{(1)}$$[^21]

Auch diese Funktion kann wiederrum kompakter formuliert werden und diese
Schreibweise wird auch für gewöhnlich verwendet:

$$\label{funktion3}
    a^{(1)}=\sigma(W\cdot a^{(0)}+b)$$

Theoretisch wenn ein Neuron einen hohen Aktivierungswert haben soll,
wenn beispielsweise eine gerade Linie erkannt wird (um mit anderen
Neuronen zusammen im späteren Verlauf aus den Mustern ganze Ziffern zu
erkennen), so müssen die Weights der zu dem Neuron führenden
Verbindungen alle möglichst niedrige Aktivierungen haben, ausser an den
Stellen an denen die Linie sich befinden soll. Um sicherzustellen, dass
es sich wirklich um eine gerade Linie handelt befindet sich direkt über
dem Strich ein Bereich in dem keine Aktivierungen sein sollten, dieser
ist rot markiert. Das erkennt man in Abbildung
[\[examples\]](#examples){reference-type="ref" reference="examples"}
sehr gut. In a erkennt man die zu erkennende Linie und in b sieht man
die zugehörigen Weights der Input Nodes zu dem Neuron. Dabei stellt grün
positive Weights da, rot negative und Weiß/Transparent ist 0. Der Bias
des Neurons stellt eine Zusätzliche Hürde oder eine Verstärkung da, was
auch in Formel [\[funktion1\]](#funktion1){reference-type="ref"
reference="funktion1"} als $b$ sichtbar ist.

### Trainieren des Neuronalen Netzwerks {#backpropagation}

Dieser Prozess ist der wichtigste. Durch das Trainieren erzielt ein
Neuronales Netzwerk den Effekt des selbstständigen Lernens. Und da die
Werte der Weights und Biases zunächst zufällig ausgewählt wurden, muss
das Netzwerk trainiert werden um nicht völligen Unsinn auszugeben.[^22]

#### Die Cost Function {#paracost}

Um herauszufinden wie gut oder schlecht ein Neuronales Netzwerk
arbeitet, also auf das Beispiel bezogen wie genau oder ungenau es
Ziffern erkennen kann, gibt es die Cost Function. Es gibt verschiedene
Arten und Möglichkeiten ähnliche Funktionen anzuwenden, hier werde ich
mich allerdings auf die Minimierung der Cost Function beziehen. Als
Ergebniss kommt eine einzige Zahl heraus welche hoch ist, wenn das
Netzwerk schlechte Ergebnisse erzielt und gegen 0 läuft, wenn das
Netzwerk sehr gute Ergebnisse liefert. Es gibt mehrere verschiedene Cost
Functions, aber ich fokussiere mich erstmal auf die MSE Funktion. MSE
steht für "Mean squared Error", sie berechnet den Cost Wert[^23] aus dem
durchschnitt der Summe der Vorhersagen und den erwarteten Ergebnissen
zum Quadrat:

$$\label{costfunction}
    MSE = \frac{1}{m} \sum^{m}_{i=1}(x^{(i)}-y^{(i)})^2$$

Wenn:

-   i = Index der Trainingsdaten

-   x = Vorhersage des Netzwerks

-   y = Erwartetes (richtiges) Ergebniss

-   m = Anzahl der Trainingsdaten[^24]

#### Gradientenabstiegsverfahren {#paragrad}

Leider ist es bei solch großen Funktionen nicht mehr möglich (stimmt
das? !TODO!) das Globale Minimum explizit zu bestimmen.

r80mm

Daher berechnet man die Steigung $\Delta C$[^25] der Funktion und
bestimmt anschließend die Richtung $-\Delta C$ in welche der Graph
sinkt. In Abbildung [\[2dcost\]](#2dcost){reference-type="ref"
reference="2dcost"} ist der Graph nur Zweidimensional und daher gibt es
nur eine Richtung in welcher der Graph fallen kann. So wird die Eingabe
immer weiter so verändert, dass sich die Cost Function minimiert. Dies
passiert in mehreren Iterationen oder auch Epochen, in welchen die
Veränderungen, also Schritte in Richtung $-\Delta C$, in Abhänigkeit von
der Steigung, immer kleiner werden um einen Überschuss zu
verhindern.[^26] Das ist auch der Grund weswegen Trainingszeiten
exponentiell zur Genauigkeit ansteigen. Die Größe dieser Schritte wird
auch Lernrate / Learning Rate genannt.[^27]

r80mm ![image](GradientDescent.png)

Bei mehrdimensionalen Funktionen, wie auch den Neuronalen Netzwerken,
gibt es mehr Möglichkeiten in welche Richtung der Graph am schnellsten
Fallen könnte. Die oben beschriebene Technik die dafür verwendet wird
nennt sich das Gradientenabstiegsverfahren. Da die Funktionen in echt
allerdings deutlich komplizierter sind, gibt es sehr viele Extrema und
da das Ziel ein möglichst tiefer Extrempunkt ist, versucht man zu
verhindern, dass man in einem solcher hohen Minima "stecken bleibt".
Deswegen verwendet man meistens abgewandelte Formen des
Gradientenabstiegsverfahrens, zum Beispiel mit einem Modifikator für
Momentum. So fällt unter Umständen "der Ball" weiter in ein tieferes
Minimum, was in Abbildung [\[3dcost\]](#3dcost){reference-type="ref"
reference="3dcost"} zu erkennen ist. Desweiteren wird für Machine
Learning fast ausschließlich "Stochastic gradient descent" verwendet, da
das Gradientenabstiegsverfahren allein viel zu aufwendig ist. Mit dieser
Form verliert man etwas Genauigkeit, der Prozess geht aber ein
vielfaches schneller vonstatten. Anstatt den Gradienten von allen
Trainingsdaten zusammen zu suchen, teilt man die Trainingsdaten in
mehrere "Batches" auf und wendet auf diesen das
Gradientenabstiegsverfahren an. So werden häufiger/schneller Schritte
richtung Extrempunkt gemacht, diese sind dafür aber ungenauer (sie
repräsentieren nicht den schnellsten Weg).[^28]

#### Ketten Regel {#parachain}

Die Kettenregel vereinfacht das Ableiten von komplizierten Funktionen
mit Logarithmen und Wurzeln, E-Funktionen und auch Klammern. Wie man
diese anwendet sieht man in Formel
[\[kettenregel\]](#kettenregel){reference-type="ref"
reference="kettenregel"}.[^29]

$$\label{kettenregel}
    \begin{matrix}
        f(x) = u(g(x))\\
        f'(x) = u'(g(x)) \cdot g'(x)
    \end{matrix}$$

Dieses Verfahren wird verwendet um die Ableitungen der Cost Function und
des Netzwerks zu finden.

#### Backpropagation {#backpropagation-1}

Jetzt ist klar was das Ziel des Trainingsprozesses ist, aber wie wird
die Cost Function minimiert? Hier kommt der Backpropagation Algorithmus
ins Spiel. In Kombination mit der Cost Function und den soeben
besprochenen Verfahren bestimmt dieser Algorithmus welche und um wie
viel die Weights und Biases des Neuronalen Netzwerks angepasst werden
müssen.

## Abwandlungen

Heute gibt es sehr viele verschiedene Abwandlungen von diesen Techniken
und auch Neuronale Netzwerke, die zwar ähnlich aufgebaut sind, die
besser für manche Zwecke sind als andere. Da gibt es zum Beispiel
"Convolutional Neural Networks"[^30], welche besonders gut zum erkennen
von Objekten in Bildern geeignet sind und daher auch eins in
[3.2](#erstellen des modells){reference-type="ref"
reference="erstellen des modells"} verwendet wird. Oder "Long short Term
memory Networks"[^31], welche speziell auf das erkennen von Stimmen
ausgelegt sind.

### Convolutional Neural Networks

Convolutional Neural Networks versuchen ein Problem der normalen
Neuronalen Netzwerke im Bereich der Bilderkennung zu lösen. Während in
einem normalen Netzwerk die Position des zu erkennenden Objektes im Bild
eine Rolle spielt, wird dies zum Großteil in einem Convolutional Neural
Network durch die veränderte Funktionsweise behoben/verbessert.

# Labelcheck als Smartphone App

In diesem Kapitel wird mithilfe von Python und TensorFlow ein Netzwerk
erstellt und trainiert, sowie anschließend eine mobile App mit Dart und
Flutter entwickelt, welche dann öffentlich für den Download bereit
stehen soll.

## Die Idee

Die Idee ist, dass die App es ermöglicht im Supermarkt die verschiedenen
Label der Produkte zu scannen und dem Nutzer dann Auskunft über die
Vertrauenswürdigkeit und generelle Aussage des Labels gibt.

Der Name bedeutet einfach nur "Label überprüfen"

## Erstellen des Models {#erstellen des modells}

Zum erstellen und trainieren des Modells werde ich die Sprache Python
und das Framework TensorFlow verwenden.

### Die Trainingsdaten

### Trainieren des Modells mit TensorFlow und Python

## Entwickeln der App

Zum entwickeln der App verwende ich die Sprache Dart und das zugehörige
Framework Flutter. Im Gegensatz zu nativ geschriebenen Apps bietet
Flutter die möglichkeit nur einmal den Code in Dart zu schreiben und
anschließend kann die App für alle großen Platformen kompiliert werden,
dazu zählen iOS, Android, aber auch Linux, Windows, MacOS und
Web/Javascript. Nun muss die App ja Zugriff auf die Kamera haben und
auch in die Supermärkte "mitgebracht" werden, weshalb nur iOS und
Android relevant sind.

### Das Framework: Flutter

Flutter Apps funktionieren anders als nativ entwickelte Apps.
Herkömmliche native Apps verwenden die UI[^32] Komponenten des
Betriebssystems und sehen daher auf jedem Gerät mit unterschiedlichen
Betriebssystemversionen leicht unterschiedlich aus. Flutter hingegen
stellt ein "Canvas" Element bereit, welches als unterliegende
Grafik-Engine Google's Skia nutzt.[^33] In Flutter stehen eine Menge UI
Komponenten zur verfügung die entweder Googles Material Design
guidelines oder Apples Human interface guidelines folgen. Der Dart Code
stellt dann als Einzigen Eintrittspunkt die `main()`{.Dart} Methode
bereit, aus welchem dann die App gestartet wird. Das Framework wird mit
der Methode `runApp(Widget)`{.Dart} initialisert. In Flutter ist jedes
UI Element ein "Widget", wodurch sich dann in Kombination in einer App
große Widgethierarchien erstellen lassen. Der Code einer simplen App,
welche nur den Text "Hello World!" in der Mitte des Bildschirms anzeigen
würde, sehe demnach so aus:

l75mm

``` {.Dart fontsize="\\footnotesize" linenos=""}
import 'package:flutter/widgets.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text('Hello World!'),
    );
  }
}
```

Zeile 1: Importieren der Widgets aus dem Framework zum bereitstellen der
Klassen, wie `Stateless`{.Dart}- und `StatefulWidget`{.Dart} und den
Methoden, wie `runApp()`{.Dart}.

Zeile 3: Die `main()`{.Dart} Methode mit dem einzigen Aufruf
`runApp()`{.Dart}, was die Klasse `MyApp`{.Dart} als Flutter App
initialisert.

Zeile 5f.: `MyApp`{.Dart} erbt die Klasse `StatelessWidget`{.Dart},
überschreibt die `build()`{.Dart} Methode und gibt eine Widgethierarchie
zurück.

Zeile 8f.: Das `Center`{.Dart} Widget nimmt als einzigen benannten
Parameter ein weiteres (child) Widget an, was in diesem Fall ein
`Text`{.Dart} Widget ist. Stateless bedeutet hier, dass sich der Zustand
des Widgets nicht während der Laufzeit verändern kann. Im Gegensatz dazu
gibt es auch noch `StatefulWidget`{.Dart}'s welche die Möglichkeit haben
bei Bedarf das UI zu "rebuilden".

Um der App das Aussehen von nativen Apps zu verleihen, verwendet man
üblicherweise ein `MaterialApp`{.Dart} (Material Design / Android) oder
`CupertinoApp`{.Dart} (Human Interface / iOS) Widget, was zudem noch
wichtige Variablen, wie `ThemeData`{.Dart} für verschiedene Farben, die
in der App einfach verwendet werden können oder
`LocalizationsDelegate`{.Dart} welche für die Bereitstellung
verschiedener Sprachen gebraucht werden, beinhaltet.

### Importieren des Models

Es gibt eine speziell für mobile Geräte angepasste Version von
TensorFlow mit dem Namen TensorFlow Lite. TFLite hat einen geringeren
Speicherbedarf kann dafür aber auch weniger als das herkömmliche
Framework.[^34] Implementationen dafür gibt es allerdings nicht in Dart,
stattdessen verwendet PlatformChannel's in Flutter, welche die
Möglichkeit bieten platformspezifischen Code aus einer Flutter App
auszuführen (Java/Kotlin für Android und Objective-C/Swift für
iOS).[^35]

Desweiteren muss das zuvor erstellte TensorFlow Model in ein TensorFlow
Lite Model umgewandelt werden.

### Veröffentlichen der App

Ich habe die App im Google PlayStore veröffentlicht, theoretisch wäre es
auch möglich die App für iOS im Appstore anzubieten, leider ist ein
Apple Entwicklerkonto mit höheren Kosten verbunden.

## Testen der App

## Fazit

Ich habe in den letzen Monaten viel über die App entwicklung mit Flutter
gelernt, leider habe ich mit dieser App schon etwas früher begonnen,
weswegen ich heute wahrscheinlich viele Dinge anders gemacht hätte. Ich
habe nicht viel darauf geachtet den UI Code von dem Logik Code zu
trennen und jetzt kommt es häufig vor, dass ich lange nach etwas suchen
muss. Ich plane allerdings dies noch zu beheben. Auch habe ich viele
Variablen als dynamisch deklariert, was ich auch noch verbessern sollte.

[\[Anhang\]]{#Anhang label="Anhang"}

# Anhang

## Weitere Aktivierungsfunktionen {#anhang:weitereaktivierungsfunktionen}

Die ReLU (Rectified linear Unit) Funktion ist im Vergleich zu anderen
Aktivierungsfunktionen, wie der Sigmoidfunktion oder der Hyperbolischen
Tangente, deutlich simpler, was sich in Leistungsansprüchen des
Trainingsprozesses wiederspiegelt.[^36]

## Code für das Beispiel aus [2.3](#funktionsweise){reference-type="ref" reference="funktionsweise"} {#anhang:colab1}

``` {.python fontsize="\\footnotesize" linenos=""}
import tensorflow as tf
import tensorflow.keras.layers as layers

numberOfNeuronsInFirstLayer = 16 
numberOfNeuronsInSecondLayer = 16 
numOfEpochs = 5 

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Laden des MNIST Datasets
# Und aufteilen in Trainigsdaten und Testdaten

x_train = tf.keras.utils.normalize(x_train, axis=1) # Normalisieren des Datasets
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # Erstellen des Neuronalen Netzwerks
model.add(layers.Flatten())
model.add(layers.Dense(numberOfNeuronsInFirstLayer, activation=tf.nn.sigmoid))
# Hinzufügen der Layer
model.add(layers.Dense(numberOfNeuronsInSecondLayer, activation=tf.nn.sigmoid))
model.add(layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # Kompilieren der Layer zu einem trainierfähigen Modell

model.fit(x_train, y_train, epochs=numOfEpochs)
# Trainieren des Modells mit den Trainingsdaten und x Epochen

model.save('beispielModel_MNIST') # Speichern des Modells
```

Ein interaktives Beispiel gibt es zusätzlich hier in meinem Colab
Notebook: <https://bit.ly/34Ggfuh>[^37]

## Die Flutter Architektur {#anhang:flutterarc}

Flutters Architektur ist in drei Ebenen unterteilt. Als Basis die
"Embedder Ebene", welche für jede Platform angepasst werden muss und zum
Beispiel für das Thread Management zuständig ist. Dadrüber liegt die
"Engine Ebene", welche zu einem Großteil in C++ geschrieben ist und zu
welcher auch die Grafikengine Skia gehört. Dadrüber liegt die "Framework
Ebene", welche zum Beispiel die UI Komponenten beinhaltet und komplett
in Dart entwickelt wird.[^38]

[^1]: [@timcookquote Tim Cook (CEO von Apple) In einem Interview mit MIT
    Technology Review]

[^2]: [@nvidiatensorcores NVIDIA Grafikprozessoren mit integrierten
    Tensor Kernen]

[^3]: Beispiel: Autonome Waffen, wie Drohnen, welche Ziele autonom
    erfassen können oder auch "Deep-fakes"

[^4]: Künstliche Intelligenz

[^5]: [@deepmindprotein]

[^6]: [@alogicalcalculus]

[^7]: [@nnhistory Absatz 3]

[^8]: [@paulwerbosbackpropagation]

[^9]: Genaueres in Kapitel [2.3](#funktionsweise){reference-type="ref"
    reference="funktionsweise"}

[^10]: Ab jetzt als CNN bezeichnet

[^11]: [@cnnhistory]

[^12]: [@geoffrey]

[^13]: [@heaton Alle drei Faustregeln]

[^14]: siehe Kapitel [2.3.1](#backpropagation){reference-type="ref"
    reference="backpropagation"}

[^15]: [@3blue1brown]

[^16]: [@googleMlGlossar] Schlüsselwort: Activation Function

[^17]: [@3blue1brown]

[^18]: siehe Anhang
    [4.1](#anhang:weitereaktivierungsfunktionen){reference-type="ref"
    reference="anhang:weitereaktivierungsfunktionen"}

[^19]: [@nnfs] Seite 76 folgende

[^20]: [@nnfs] Seite 185

[^21]: Gleichungen [\[funktion2\]](#funktion2){reference-type="ref"
    reference="funktion2"} und
    [\[funktion3\]](#funktion3){reference-type="ref"
    reference="funktion3"} von [@3blue1brown]

[^22]: Ein Code Beispiel, wie man ein solches Netzwerk mit modernen
    Frameworks erstellen und trainieren würde befindet sich im Anhang
    [4.2](#anhang:colab1){reference-type="ref"
    reference="anhang:colab1"}.

[^23]: Manchmal auch Loss genannt, meint das gleiche.

[^24]: Formel [\[costfunction\]](#costfunction){reference-type="ref"
    reference="costfunction"} und Erklärung vergleiche [@towardsds]

[^25]: $C$ bezieht sich hier auf die **C**ost function

[^26]: [@3blue1brown]

[^27]: [@readthedocsgradientdescent]

[^28]: [@3blue1brown] und [@mitstochasticgd]

[^29]: [@kettenregel]

[^30]: auch CNN; deutsch: **faltendes** neuronales Netzwerk

[^31]: auch LSTMN; deutsch: Langes **Kurzzeitgedächtnis** Netzwerk

[^32]: User Interface; deutsch: Benutzeroberfläche

[^33]: [@flutterarchitecture]; mehr dazu im Anhang
    [4.3](#anhang:flutterarc){reference-type="ref"
    reference="anhang:flutterarc"}

[^34]: [@tflite]

[^35]: [@flutterplatformcode]

[^36]: [@nnfs]

[^37]: Ungekürzter Link:
    <https://colab.research.google.com/drive/1ty_QQlL038YT6KpBjSdqGvIGyH0YXwxW>

[^38]: [@flutterarchitecture]
