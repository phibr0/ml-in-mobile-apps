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
Google's Tochterfirma DeepMind im Dezember 2020 eine KI[^4] präsentiert,
welche das Falten von Proteinen akkurat prognostizieren kann; dies war
vorher nur sehr langsam und deutlich ungenauer möglich.[^5]

## Ziel der Arbeit

Mein persönliches Ziel ist es, mehr über den Aufbau von Neuronalen
Netzen und die Funktionsweise von Machine Learning zu lernen. Außerdem
möchte ich auch ein praktisches Ergebniss haben, dafür habe ich im
Kapitel !TODO! eine App entwickelt, welche dem Nutzer mehr Informationen
über Produkte beim einkaufen liefern soll.

# Neuronale Netzwerke

## Geschichte

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

### Erstellung eines Neuronalem Netzwerks anhand eines Beispiels

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
Trainingsprozess[^14] angepasst.

``` {.python fontsize="\\scriptsize" linenos=""}
    import tensorflow as tf
        
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()

    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])
    
```

## Funktionsweise

r75mm

Ein Neuronales Netzwerk kann man sich eigentlich als eine große
Mathematische Funktion vorstellen. In dem zuvor genannten Beispiel wäre
es eine Funktion mit 576 Variablen und 10 Ergebnissen. Gibt man dieser
Funktion nun ein Bild, beziehungsweise 576 Werte als Input, so werden
von links nach rechts alle Weights $w$ und Biases $b$ zusammen mit dem
vorigen Aktivierungswerten $a$ berechnet. Da ein Neuron aber nur Werte
im Bereich $0\leq x \leq 1$ haben kann, so wird das Ergebniss noch
mithilfe einer weiteren Funktion in diesen Bereich umgewandelt. Eine
Häufig verwendete Funktion ist dabei die Sigmoidfunktion, siehe
Abbildung [\[sigmoid\]](#sigmoid){reference-type="ref"
reference="sigmoid"}.[^15]

$$\sigma(w_1a_1+w_2a_2+w_3a_3+\cdot \cdot \cdot +w_na_n-b)$$

Theoretisch wenn ein Neuron einen hohen Aktivierungswert haben soll,
wenn beispielsweise eine gerade Linie erkannt wird (um mit anderen
Neuronen zusammen im späteren Verlauf aus den Mustern ganze Ziffern zu
erkennen), so müssen die Weights der zu dem Neuron führenden
Verbingungen alle möglichst kleine Werte haben, ausser an den Stellen an
denen die Linie sich befinden soll. Um sicherzustellen, dass es sich
wirklich um eine gerade Linie handelt befindet sich direkt über dem
Strich ein Bereich in dem keine Aktivierungen sein sollten, dieser ist
rot markiert. Das erkennt man in Abbildung
[\[examples\]](#examples){reference-type="ref" reference="examples"}
sehr gut. In a erkennt man die zu erkennende Linie und in b sieht man
die zugehörigen Weights der Input Nodes zu dem Neuron. Dabei stellt grün
positive Weights da, rot negative und Weiß/Transparent ist 0. Der Bias
des Neurons stellt eine Zusätzliche Hürde oder eine Verstärkung da.

### Trainieren - Backpropagation {#backpropagation}

[^1]: [@timcookquote Tim Cook (CEO von Apple) In einem Interview mit MIT
    Technology Review]

[^2]: [@nvidiatensorcores NVIDIA Grafikprozessoren mit integrierten
    Tensor Kernen]

[^3]: Beispiel: Autonome Waffen, wie Drohnen, welche Ziele autonom
    erfassen können

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
