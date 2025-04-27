# Convolutional Neural Network - MNIST & Simple Objects

Projekt napisany w języku Python tworzący sieć konwolucyjną (CNN) do rozpoznawania podstawowych cyfr pisanych odręcznie oraz prostych zdjęć przedmiotów.  
Biblioteka **TensorFlow** jest używana wyłącznie do wczytywania danych MNIST.

## Funkcjonalności

- Wczytywanie zbioru danych MNIST przy użyciu TensorFlow.
- Budowanie i trenowanie własnej sieci konwolucyjnej.
- Klasyfikacja obrazów cyfr oraz prostych przedmiotów.
- Automatyczne testowanie przetrenowanego modelu na zbiorze testowym.
- Wyświetlanie wyników przypisania klas do obrazów.
- Wizualizacja błędnie sklasyfikowanych przykładów w osobnym oknie.

## Wymagania

Aby uruchomić projekt, wymagane są następujące biblioteki:

- `tensorflow`
- `numpy`
- `matplotlib`

Możesz je zainstalować za pomocą polecenia:

```bash
pip install tensorflow numpy matplotlib
```

## Uruchomienie

Po zainstalowaniu wymaganych bibliotek uruchom skrypt:

```bash  
python main.py
```

## Dane

Projekt wykorzystuje zbiór danych MNIST:

- Zbiór składa się z obrazów przedstawiających cyfry odręczne (0-9).
- Dane są ładowane automatycznie przy użyciu TensorFlow.
- Możliwa jest rozbudowa o własne zdjęcia prostych przedmiotów (wymaga dodatkowego przygotowania danych).

## Architektura Modelu

Model własnoręcznie zaimplementowanej sieci konwolucyjnej (CNN) zawiera:

- **Wejście**: pojedynczy obraz o wymiarach 28x28 pikseli (np. cyfra MNIST).
- **Warstwa 1**:
  - Splot (`convolve2D`) za pomocą własnych filtrów.
  - Aktywacja funkcją ReLU (`relu`).
  - Normalizacja BatchNorm (`batch_norm`).
  - Operacja MaxPooling (`max_pooling`) o rozmiarze 2x2.
- **Warstwa 2**:
  - Ponowne wykonanie operacji konwolucji na wyjściu z pierwszej warstwy.
  - MaxPooling.
- **Spłaszczenie**:
  - Przekształcenie map cech w pojedynczy wektor (`flatten`).
- **Warstwa w pełni połączona 1**:
  - Wektor cech przechodzi przez w pełni połączoną warstwę (`fully_connected`) o 128 neuronach.
  - Aktywacja funkcją ReLU.
- **Warstwa w pełni połączona 2 (wyjściowa)**:
  - Druga w pełni połączona warstwa (`fully_connected`) o 10 wyjściach odpowiadających klasom (0-9).
  - Softmax (`softmax`) przekształcający wyniki w prawdopodobieństwa.

## Proces uczenia

- Strata obliczana przy użyciu funkcji **Cross-Entropy** (`cross_entropy_loss`).
- Aktualizacja wag za pomocą własnej implementacji **Backpropagation** (`backpropagation`).
- Parametry aktualizowane metodą Gradient Descent.

## Proces testowania

- Przejście danych testowych przez wytrenowany model (`forward2`).
- Porównanie przewidywanych etykiet z rzeczywistymi.
- Obliczenie dokładności klasyfikacji.
- Wyświetlenie przykładowych błędnie sklasyfikowanych obrazów (z przewidywaną i prawdziwą klasą).

## Podsumowanie

Model jest w pełni zaimplementowany "od podstaw" (bez użycia wysokopoziomowych bibliotek typu Keras/PyTorch), co pozwala na pełne zrozumienie działania sieci konwolucyjnych krok po kroku.
