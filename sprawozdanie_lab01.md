# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy  
**Zajęcia nr / temat:** Laboratorium 01 — Sekwencje i sygnał mowy: od fali dźwiękowej do reprezentacji  
**Autor / grupa:** Bartosz Paluda
**Data:** 21.10.2026  

---

## 1. Cel ćwiczenia

Celem laboratorium było zbadanie procesu przekształcania ciągłego sygnału mowy w reprezentacje macierzowe stanowiące wejście dla głębokich modeli sekwencyjnych (sieci rekurencyjnych, transformatorów i konwolucyjnych sieci neuronowych). 

W ramach ćwiczenia zrealizowano:
1. Wczytanie i resampling trzech nagrań o różnym stopniu złożoności (izolowana samogłoska, pojedynczy wyraz, mowa ciągła — całe zdanie) do standardu $16\text{ kHz}$ mono.
2. Wyznaczenie i wizualizację trzech reprezentacji: fali dźwiękowej (przebiegu czasowego), spektrogramu log-mel ($64$ pasma mel) oraz współczynników cepstralnych MFCC ($13$ współczynników).
3. Przeprowadzenie eksperymentów badających wpływ długości okna analizy (`n_fft` $\in \{256, 512, 1024\}$) oraz kroku ramkowania (`hop_length` $\in \{80, 160, 320\}$) na spektrogram, z weryfikacją kompromisu czasowo-częstotliwościowego (zasada nieoznaczoności Gabora).
4. Analizę cech akustycznych widocznych w poszczególnych reprezentacjach w odniesieniu do zjawisk fonetycznych (formanty, harmoniczne, wybuchy zwarć, pauzy, koartykulacja).

---

## 2. Metoda i implementacja

### Narzędzia i parametryzacja bazowa
Eksperymenty przeprowadzono w środowisku Google Colab w języku Python przy użyciu bibliotek `librosa`, `numpy` oraz `matplotlib`. 

Konfiguracja bazowa przetwarzania:
- **Częstotliwość próbkowania (`SR`):** $16000\text{ Hz}$ (granica Nyquista wynosi $f_{\text{Nyquist}} = 8000\text{ Hz}$), pojedynczy kanał (mono).
- **Długość okna FFT (`N_FFT`):** $512$ próbek ($32\text{ ms}$ przy $16\text{ kHz}$, rozdzielczość częstotliwościowa $\Delta f = 31,25\text{ Hz}$), funkcja okna Hanna minimalizująca przeciek widmowy.
- **Krok ramkowania (`HOP_LENGTH`):** $160$ próbek ($10\text{ ms}$, co daje nakładanie sąsiednich ramek na długości $352$ próbek / $22\text{ ms}$ oraz generuje $100$ ramek na sekundę).
- **Bank filtrów mel (`N_MELS`):** $64$ trójkątne filtry o zagęszczonym rozstawie w niskich częstotliwościach i logarytmicznym w wyższych.
- **Skala logarytmiczna:** przeliczenie mocy do decybeli za pomocą `librosa.power_to_db` z referencją `ref=np.max` ($0\text{ dB}$ odpowiada punktowi o maksymalnej energii w nagraniu, a dolna granica dynamiki wynosi $-80\text{ dB}$).
- **Liczba współczynników MFCC (`N_MFCC`):** $13$ (wyliczanych za pomocą DCT typu II z wcześniej obliczonej macierzy log-mel, co zapewnia wspólną bazę filtrów mel i identyczną siatkę czasową).

### Kod przetwarzania

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
N_MFCC = 13

def wczytaj_audio(sciezka):
    y, sr = librosa.load(sciezka, sr=SR, mono=True)
    return y.astype(np.float32), sr

def oblicz_reprezentacje(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, n_mfcc=N_MFCC):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    M = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)
    return S_db, M
```

---

## 3. Wyniki

### Zestawienie wymiarów reprezentacji
Dla zarejestrowanych nagrań 

| Nagranie | Czas trwania [s] | Liczba próbek $N$ | Kształt Log-mel `[cechy, T]` | Kształt MFCC `[cechy, T]` | Układ dla RNN/Transformer `[T, cechy]` |
|---|---|---|---|---|---|
| **1. Samogłoska** | $1,42\text{ s}$ | $22\ 720$ | `[64, 143]` | `[13, 143]` | `[143, 64]` / `[143, 13]` |
| **2. Wyraz** | $0,71\text{ s}$ | $11\ 360$ | `[64, 72]` | `[13, 72]` | `[72, 64]` / `[72, 13]` |
| **3. Zdanie** | $5,27\text{ s}$ | $84\ 320$ | `[64, 527]` | `[13, 527]` | `[527, 64]` / `[527, 13]` |

*Wygenerowane wykresy:*
- <img width="1500" height="1200" alt="samogłoska_wykres" src="https://github.com/user-attachments/assets/e09c291b-52bd-4d36-b501-559359057d49" />

- <img width="1500" height="1200" alt="wyraz_wykres" src="https://github.com/user-attachments/assets/6e008c3a-c89e-4a11-92c9-1e64e13e8b5d" />

- <img width="1500" height="1200" alt="zdanie_wykres" src="https://github.com/user-attachments/assets/23767b5e-e2d4-4de3-9168-721a3711282e" />


---

### Eksperyment A: Wpływ długości okna (`n_fft`) na spektrogram zdania
Badanie przeprowadzono przy stałym kroku `hop_length = 160` ($10\text{ ms}$) dla wartości `n_fft`: $256$ ($16\text{ ms}$), $512$ ($32\text{ ms}$) oraz $1024$ ($64\text{ ms}$).

* **Hipoteza:** Zwiększenie parametru `n_fft` wydłuża czas analizowanego wycinka, co polepsza rozdzielczość częstotliwościową (mniejsze $\Delta f$), uwidaczniając poszczególne prążki harmoniczne samogłosek, kosztem rozmycia szybkich zmian czasowych (gorsza rozdzielczość czasowa).
* **Obserwacja z wykresu `eksperyment_A.png`:**
  - `n_fft = 256` ($\Delta f = 62,50\text{ Hz}$): Pionowe granice wybuchów i przejść między głoskami (np. w okolicach $0,35\text{ s}$, $1,55\text{ s}$ i $3,10\text{ s}$) są ostre i precyzyjnie zlokalizowane. W osi pionowej harmoniczne zlewają się w jednolite, szerokie pasma formantowe.
  - `n_fft = 1024` ($\Delta f = 15,62\text{ Hz}$): W rejonach samogłoskowych (zwłaszcza w paśmie $0\text{–}2000\text{ Hz}$ w przedziale $0,50\text{–}0,75\text{ s}$) widoczne są wyraźne, odseparowane poziome prążki harmonicznych głosu. Zjawiska transjentowe (nagłe impulsy i zwarcia) uległy jednak poziomemu rozmyciu na szerokość $64\text{ ms}$.
  - `n_fft = 512` ($\Delta f = 31,25\text{ Hz}$): Zapewnia optymalny kompromis, oddając zarówno formanty samogłosek, jak i dostatecznie ostre krawędzie fonemów.
<img width="1500" height="1200" alt="eksperyment_A" src="https://github.com/user-attachments/assets/affca5a3-8b9d-41a3-8856-2d35dad6e73c" />

---

### Eksperyment B: Wpływ kroku ramkowania (`hop_length`) na spektrogram zdania
Badanie przeprowadzono przy stałej długości okna `n_fft = 512` ($32\text{ ms}$) dla wartości `hop_length`: $80$ ($5\text{ ms}$), $160$ ($10\text{ ms}$) oraz $320$ ($20\text{ ms}$).

* **Hipoteza:** Krok ramkowania decyduje wyłącznie o gęstości próbkowania w osi czasu (liczbie generowanych kolumn $T$), nie wpływając na fizyczną rozdzielczość częstotliwościową pojedynczego okna.
* **Obserwacja z wykresu `eksperyment_B.png`:**
  - `hop_length = 80`: Generuje $T = 1054$ ramek. Spektrogram jest wygładzony i ciągły w osi poziomej.
  - `hop_length = 160`: Generuje $T = 527$ ramek (standardowe $100\text{ ramek/s}$). Zapewnia pełną czytelność zjawisk fonetycznych.
  - `hop_length = 320`: Generuje $T = 264$ ramki. Obraz ulega widocznej pikselizacji (dyskretyzacji schodkowej w poziomie), a krótkie zjawiska fonetyczne (np. wybuchy zwarć) są reprezentowane przez zaledwie 1–2 ramki. Rozstaw prążków w pionie pozostaje we wszystkich trzech przypadkach identyczny.
<img width="1500" height="1200" alt="eksperyment_B" src="https://github.com/user-attachments/assets/f0c469dd-a962-46aa-a0e3-8719a9ce78f4" />

---

## 4. Analiza i interpretacja

### 1. Stopniowanie złożoności sygnałów (Samogłoska $\rightarrow$ Wyraz $\rightarrow$ Zdanie)
- **Izolowana samogłoska (`samogłoska_wykres.png`):**
  - Sygnał wykazuje charakter niemal stacjonarny w przedziale $0,05\text{–}1,25\text{ s}$.
  - Na spektrogramie log-mel dominuje stały układ prążków harmonicznych oraz rezonansów traktu głosowego (formantów) skoncentrowanych w pasmach poniżej $2000\text{ Hz}$ (główne wzmocnienia w rejonie $300\text{–}800\text{ Hz}$ oraz $1100\text{–}1500\text{ Hz}$).
  - Wykres MFCC jest niemal jednorodny wzdłuż osi czasu (współczynniki wyższych rzędów 1–12 utrzymują stałe wartości), co dowodzi braku zmian kształtu obwiedni traktu głosowego w trakcie fonacji.
- **Wyraz z fonemem zwartym (`wyraz_wykres.png`):**
  - W przedziale $0,00\text{–}0,23\text{ s}$ występuje cisza przedartykulacyjna (brak wychylenia na fali, spektrogram na poziomie $-80\text{ dB}$, współczynnik $0$ MFCC na poziomie poniżej $-600$).
  - W chwili $t = 0,26\text{ s}$ występuje charakterystyczna, bardzo wąska pionowa linia o dużej energii obejmująca pełne pasmo do $8192\text{ Hz}$ — odpowiada ona wybuchowi zwarcia (plozji) spółgłoski zwartej.
  - Od chwili $t = 0,27\text{ s}$ do $0,67\text{ s}$ pojawia się periodyczny sygnał samogłoski z wyraźnymi formantami w zakresie do $2500\text{ Hz}$.
- **Mowa ciągła (`zdanie_wykres.png`):**
  - Widoczna jest wysoka dynamika zjawisk akustycznych. Formanty nie tworzą linii prostych, lecz dynamicznie falują w osi częstotliwości, odzwierciedlając koartykulację (ruch narządów mowy pomiędzy kolejnymi głoskami).
  - Występują wyraźne pauzy międzywyrazowe (np. w przedziałach $1,15\text{–}1,30\text{ s}$, $2,60\text{–}2,85\text{ s}$ oraz $3,60\text{–}3,80\text{ s}$), widoczne jako spadki energii do $-80\text{ dB}$ we wszystkich pasmach.
  - Wykres MFCC w mowie ciągłej charakteryzuje się dużą zmiennością w czasie i traci bezpośrednią czytelność wizualną; współczynnik $0$ wiernie odzwierciedla kontur energii (wartości silnie ujemne w pauzach i podwyższone podczas fonacji).

---

## 5. Wnioski

1. **Drastyczna redukcja wymiarowości sekwencji:**  
   Przekształcenie fali dźwiękowej do spektrogramu log-mel zredukowało liczbę wartości dla 5-sekundowego zdania z $84\ 320$ próbek do macierzy $64 \times 527$ ($33\ 728$ wartości), a dla MFCC do $13 \times 527$ ($6\ 851$ wartości). Reprezentacja cech staje się zwarta, usuwa zbędną fazę sygnału i skupia się na informacjach percepcyjnych istotnych dla rozpoznawania mowy.
2. **Potwierdzenie zasady nieoznaczoności Gabora:**  
   Dobór parametru `n_fft` zawsze stanowi kompromis pomiędzy rozdzielczością czasową a częstotliwościową. Do analizy mowy przy $\text{SR} = 16\text{ kHz}$ wartość `n_fft = 512` ($32\text{ ms}$) okazuje się optymalna, ponieważ czas trwania okna jest krótszy od typowej długości fonemu ($50\text{–}150\text{ ms}$), a jednocześnie obejmuje wystarczającą liczbę okresów tonu krtaniowego, by odwzorować strukturę formantową.
3. **Rola kroku ramkowania (`hop_length`):**  
   Zmniejszenie `hop_length` zwiększa liczbę kroków czasowych $T$, co daje gęstszą reprezentację dla sieci neuronowej, lecz nie poprawia separacji prążków częstotliwościowych. Krok $10\text{ ms}$ (`hop_length = 160`) zapewnia wystarczającą gęstość próbkowania bez nadmiernego narzutu obliczeniowego.
4. **Log-mel a MFCC w architekturach Deep Learning:**  
   MFCC poprzez dekolerację za pomocą DCT silnie kompresuje obwiednię widmową do 13 cech, co było kluczowe w klasycznych modelach HMM/GMM. Jednak współczesne głębokie sieci neuronowe (CNN, Conformer, Whisper) preferują pełniejszy spektrogram log-mel ($64$ lub $80$ pasm), ponieważ same są w stanie wyuczyć się optymalnych filtrów i korelacji przestrzenno-czasowych bez strat informacji wprowadzanych przez obcięcie DCT.
```

