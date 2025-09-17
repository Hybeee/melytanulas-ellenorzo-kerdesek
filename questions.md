# 1. Előadás
### 1. Mi a mélytanulás? Miért mély?
A mélytanulás a gépi tanulás egyik formája, melynek során neurális hálózatok segítségével végzünk el egy feladatot (pl.: regresszió/osztályozás). 

A "mély" jelző a hálók méretére utal; ezen modellek rétegekből állnak, ahol az egyes rétegek az adatok más-más reprezentációjának felelnek meg. Ezen reprezentációk a mélyebb rétegek felé haladva egyre komplexebbek és például egy osztályozási feladat esetén egyre inkább szeparálják a különböző osztályokba tartozó adatpontokat. A rétegek száma sokszor több száz, vagy akár több ezer is lehet. Illetve fontos kiemelni, hogy bizonyos architektúrák esetén az egyes rétegekben elhelyezkedő neuronok száma is megnő - ezt nevezik a háló szélességének. Ugyanakkor a mélység és szélesség nem feltétlen egyenesen arányos.
### 2. Mi a mélytanulás kapcsolata a gépi tanulással?
A mélytanulás a gépi tanulásnak egy alfaja. Úgy lehet elképzelni, hogy a mélytanulás a bogár, míg a gépi tanulás a rovar. Minden bogár rovar, de nem minden rovar bogár. 

A mély tanulás a gépi tanulásból nőtte ki magát, leginkább akkor érdemes használni, amikor sok adat áll rendelkezésünkre, illetve az adatok közötti kapcsolatok nem lineárisak. Ezen kívül a gépi tanulással szemben - ahol manuálisan kell kiválasztani - a mély tanulás automatikusan választja ki a probléma megoldását tekintve releváns jellemzőket. A két megközelítés hasonló problémákat old meg, de a megoldás megvalósításában különböznek.
### 3. Mi a szimbolikus (szabály alapú) MI és a gépi tanulás kapcsolata?
A szimbolikus MI (nás néven szabályalapú MI) előre definiált szabályokra és logikai következtetésekre épül, azaz a rendszert a fejlesztő által meghatározott szabályok vezérlik.

Ezzel szemben a gépi tanulás adatvezérelt: a rendszer magától tanul mintákat az adatokból, anélkül, hogy explicit szabályokat adnánk meg.

Mindkét megközelítés a mesterséges intelligencia része, és bizonyos feladatoknál kiegészíthetik egymást, amit a **neuro-szimbolikus AI** próbál ötvözni.
### 4. Mi a túl- és alulillesztés? (overfitting, underfitting)
A tanítás eredményeként megfigyelhető jelenségek.

Underfitting: A modell nem volt képes megtanulni az adatok közötti összefüggéseket, így rossz performanciát mutat mind a tanító, mind a validációs/teszt adathalmazon. Ennek oka általában az, hogy a modell komplexitása nem elegendően nagy és/vagy a tanítóminták száma nem elegendő. 

Overfitting: A modell túltanulásának jelensége azaz az, amikor a modell szinte tökéletesen rátanult a tanítómintákra. Ennek eredményeképpen a modell általánosítóképessége romlik, így egy teljesen új minta esetén rossz teljesítményt mutat. Fontos kiemelni, hogy ilyenkor általában a validációs, illetve teszt adathalmazokon mutatott teljesítménye is rossz - kivéve akkor, ha a tanítóminta elég reprezentatív/"végtelen" sok, lásd: double-descent jelensége.
### 5. Mire jó a hibafüggvény?
Olyan matematikai függvény, amely segítségével megvizsgálhatjuk a modell kimenetének minőségét.

A függvénynek két bemenete van, a modell kimenete, illetve az elvárt kimenet. A függvény által visszaadott érték numerikus, amely azt mutatja, hogy mennyire tér el a modell kimenete az elvárt kimenettől.

A hibafüggvény helyes megválasztása kulcsfontosságú, hiszen tanítás során arra törekszik a modell, hogy ezt minimalizálja a súlyai változtatásával. Így a hibafüggvény segítségével irányítható a modell tanulása - pl.: kevesebbszer vétsen FP, mint FN hibát.

Fontos megjegyezni, hogy különböző feladatokra - pl.: regresszió, bináris vagy többosztályos klasszifikáció - különböző hibafüggvényeket kell használni.
### 6. Mi az a tenzor, mik a fő tulajdonságai? Milyen rangú tenzor szükséges a különböző adattípusokhoz?
A **tenzor** egy matematikai objektum, amely általánosítja a skalárokat, vektorokat és mátrixokat több dimenzióra. Egyszerűen fogalmazva: egy tenzor **többdimenziós tömb**, amely számokat tárol.

**Fő tulajdonságai:**
- **Rang(rank/order):** a dimenziók számát jelenti.
    - 0-rangú tenzor: skalár (pl.: `5`)
    - 1-rangú tenzor: vektor (pl.: `[1, 2, 3]`)
    - 2-rangú tenzor: mátrix (pl.: `[[1, 2], [3, 4]]`)
    - 3-rangú és nagyobb tenzor: többdimenziós tömb (pl.: képadatok: `height x width x channel`)
- **Alak (shape):** a tenzor dimenzióinak mérete, pl.: egy `28 x 28 x 3` kép.
- **Elemtípus:** a tenzorban tárolt adatok típusa (pl.: `float32`, `int64`)
### 7. Ismertesse a neuronhálókban alkalmazott alapvető vektor és mátrixműveleteket, és az ezekhez kapcsolódó feltételeket.
A neurális hálózatok számításai nagyrészt **vektor- és mátrixműveleteken** alapulnak:
**Alapvető műveletek:**
- **Vektor-mátrix szorzás:** `input_vector * weight_matrix`
    - Feltétel: `input_vector.shape[0] == weight_matrix.shape[1]`
- **Mátrix-mátrix szorzás:** `batch_input * weight_matrix`
    - Feltételek: `batch_input.shape[1] == weight_matrix.shape[0]`
- **Elemről elemre műveletek:** `arr_1 + arr_2`
    - Feltételek: a tenzorok alakja vagy broadcastolható legyen.
- **Transzponálás:** gyakran a dimenziók összehangolására, pl.:gradiens számításnál.
### 8. Aktivációs függvényekre miért van szükség a neurális hálózatokban?
Az aktivációs függvények segítségével viszünk nemlinearitást a modellbe. Erre azért van szükség, mert valós adatok esetén nagyon ritka, hogy az adatok közötti összefüggések lineárisak lennének. Ugyanakkor a modellek működését megvizsgálva látszik, hogy az aktiváció függvény nélkül a modell kimenete csupán a bemeneti értékek egy lineáris kombinációja lenne.
### 9. Mi az az SGD (Stochastic Gradient Descent)?
A Stochastic Gradient Descent a modell tanítására használt optimalizáló algoritmus. Segítségével dönthetjük el, hogy az egyes súlyokat milyen mértékben változtassuk meg  úgy, hogy a modell hibája csökkenjen.

Jellegzetessége, hogy az egyes súlyokhoz tartozó hibákat/gradienseket pontosan egy darab - véletlenszerűen választott - tanítóminta kiértékelését követően számolja ki. Ez egyébként általában hátrányos, hiszen egy tanítóminta nem feltétlen elegendően reprezentatív a tanítóadathalmazt tekintve, így zajossá teheti a tanítást. Ugyanakkor mivel csak egy adatpontra számolja ki a gradienst, így kevésbé költséges. Ezért, ha elegendő számítási erőforrás áll rendelkezésre, általában a Mini Batch Gradient Descent használata javasolt, melynek során M darab tanítómintára számolja ki az algoritmus a gradienst - ahol M < N, N a tanítóminták száma.
### 10. Mi az a backpropagation eljárás?
A **backpropagation** a mély tanulás egyik központi és legalapabb algoritmusa. Segítségével kiszámolható, hogy a modell bizonyos súlyai a hibafüggvény értékét mennyire befolyásolták.

Az algoritmus a matematikai deriválás láncszabályán alapul, futása során a hibát visszaterjeszti az egyes súlyokhoz rétegről rétegre. A backpropagation segítségével az optimalizáló algoritmusok (például SGD) frissíthetik a háló súlyait a hibafüggvény minimalizálása érdekében.

# 2. Előadás
## 2.1 Deep Learning hardware
### 1. Mi a különbség a játékos és professzionális célú GPU-k között?
A professzionális GPU-ok célja az, hogy megbízhatóak és pontosak legyenek - pl.: ECC memória (hibajavító) -, illetve bizonyos területen speciális támogatást nyújtsanak, pl.: AI/DL. Általában nagyobb memóriakapacitással rendelkeznek.

A játékos célú GPU-k általában a nyers teljesítményt veszik figyelembe, az a cél, hogy pl.: minél nagyobb legyen az FPS egy játék esetén. Általában olcsóbbak is, mint egy professzionális GPU, de emiatt például kevésbé stabilak is. Ezek inkább grafikus teljesítményre vannak szabva, mintsem numerikus stabilitásra.
### 2. Mi a GPU fő előnye? Bit felbontás, magok száma, órajel, számítási egységek, NVLink fogalmak.
A GPU-k előnye például egy CPU-val szemben abban rejlik, hogy képesek műveletek párhuzamos elvégzésére, amely felgyorsítja a nagy mátrixműveletek elvégzését. Ez például DL esetén jelentős, hiszen egy GPU képes egy egész batch-et párhuzamosan feldolgozni.

**Fogalmak:**
- **Bit felbontás:** Éebegőpontos számítások pontossága.
- **Magok száma:** A párhuzamos számításra képes processzormagok száma.
- **Órajel:** A GPU magok működési sebessége, meghatározza, hány műveletet képes elvégezni másodpercenként.
- **Számítási egységek:** Az a hardverkomponens, amely végrehajtja a számításokat. (pl.: CUDA cores)
- **NVLink:** NVIDIA gyors kommunikációs protokoll a GPU-k és a CPU vagy a GPU-k közti adatátvitelhez, gyorsabb, mint a PCIe.
### 3. Mik a legfőbb előnyei egy professzionális GPU szervernek?
A legfőbb előnyök a következők:
- **Több GPU párhuzamos használata:** Nagyobb modellek és batch-ek gyors feldolgozáshoz.
- **Nagy memória:** Segíti a nagy modellek és adatbatch-ek kezelését.
- **Stabilitás és pontosság:** ECC memória és megbízható hardver biztosítja a hibamentes számításokat.
- **Optimalizált szoftvertámogatás:** `CUDA`, `cuDNN`, AI/DL könyvtárok maximális teljesítményhez.
- **Hatékony hűtés és energiafelhasználás:** Stabil működés nagy terhelés mellett.
### 4. Milyen tároló egységek jöhetnek szóba egy nagyteljesítményű GPU szerverben?
A következő tároló egységek jöhetnek szóba:
- **HDD:** Nagy kapacitás, olcsó, lassabb.
- **SSD:** Gyorsabb, közepes kapacitás.
- **NVMe:** Nagyon gyors, alacsony késleltetés, ideális nagy adatmennyiséghez DL feladatoknál.
- **RAID tömbök:** Adatbiztonság és/vagy gyorsabb I/O.
### 5. Mik az alkotóelemei egy több GPU szervert tartalmazó MI infrastruktúrának?
Az alábbi alkotóelemek
- **GPU szerverek:** Több GPU-val, nagy memória- és számtási kapacitással.
- **Hálózat:** Nagy sávszélességű, alacsony késleltetésű kapcsolat (pl.: `NVLink`, In`finiBand).
- **Tárolórendszer:** Gyors adateléréshez SSD/NVMe tömbök.
- **Kezelő- és menedzsment szoftverek:** Orchestration (pl.: `Kubernetes`), driver és könyvtárkezelés (`CUDA`, `cuDNN`).
- **Energia- és hűtésrendszer:** Stabil működés nagy terhelés mellett.
### 6. Mik a előnyei és hátrányai a játékos célú GPU-k deep learning célokra való használatának?
Az első kérdésre adott válasz már bemutatta a különbséget a két GPU között. Az előny ilyenkor az, hogy ha nincs professzionális GPU-ja a fejlesztőnek, akkor használhatja a játékos GPU-t is modellek tanítására, ezzel felgyorsítva a folyamatot. Ugyanakkor a játékos GPU-k pontatlanabbak, mint a professzionális GPU-k, így ebben ez kihathat a modell által tanult súlyokra, így a performanciára is. Azaz játékos GPU használatánák fennáll annak az esélye, hogy nem egy optimális modellt kap a fejlesztő.

## 2.2 Deep Learning software 
### 1. Mi a konténerizáció? Miért előnyös Deep Learning rendszerek esetén a konténerizáció?
A konténerizáció az a folyamat, melynek során egy szoftvercsomagot egy konténerbe csomagolunk. Ebben a konténerben a szoftvercsomaghoz tartozó összes függőség rendelkezésre áll, illetve önmagában a konténerben fog futni a szoftver a többi processztől függetlenül. Előnye a virtualizációval szemben, hogy nem egy teljes OS-t emulál, hanem az OS gazda kerneljét használja, így könnyű és gyors tud lenni.

A DL esetén a következő okokból előnyös:
- Függőségi konfliktusok (`Dependency Hell`)
- Eltérő környezetek (pl.: `dev` vs. `prod`)
- Nehézkes csapatmunka
- Rossz reprodukálhatóság
### 2. Mik az elsődleges mélytanuló keretrendszerek?
- **Pytorch:** Dinamikus számítási gráf, könnyen debugolható, népszerű kutatói körökben.
- **Tensorflow:** Statikus számítási gráf, ipari környezetben elterjedt, támogatja a deployment-et szerverre és mobilra.
### 3. Mi az az MLOps? Mik a fő komponensei?
Az `MLOps`, másnéven `Machine Learning Operations` a gépi tanulási modellek fejlesztésének, telepítésének és üzemeltetésének folyamatait standardizáló és automatizáló gyakorlat.

**Fő komponensei:**
- **Verziókezelés:** Kód, adat és modellek követése.
- **CI/CD pipeline:** Automatikus build, teszt és deploy folyamatok.
- **Monitoring:** Modell teljesítményének, drift és hibák követése.
- **Automatizált training és retraining:** Modell frissítése új adatok alapján.
- **Infrastruktúra menedzsment:** Konténerek, GPU/CPU erőforrások kezelése.
### 4. Mi az a SLURM, Kubernetes és Docker konténerizáció? Melyiket, mikor használjuk?
- **Docker:** Konténerizációs platform, amely izolált környezetben futtatja a szoftvereket. Fejlesztéskor, teszteléskor és kisebb GPU/CPU környezetekben használt.
- **Kubernetes:** Konténer-orchestration rendszer, amely több Docker konténer menedzselését és skálázását teszi lehetővé. Használjuk nagyléptékű, több szerveres, production környezetben.
- **SLURM:** Cluster menedzsment és ütemező HPC (`High Performance Computing`) környezetekhez. Nagy számításigényű feladatokhoz használt, pl.: több GPU-s deep learning modellek futtatására klaszteren.

## 2.3 Deep Learning alapú osztályozás és regresszió
### 1. Mi az a regresszió? Mondj pár példát regressziós feladatra.
A regressziós feladat célja, hogy bemeneti jellemzők alapján egy folyamatos numerikus értéket becsüljön a modell. Pl.: Ház jellemzői alapján a ház ára.
### 2. Mi az az osztályozás? Mondj pár példát osztályozási feladatra.
Az osztályozási feladat célja, hogy a bemeneti jellemzők alapján a bemeneti adatpontot adott osztályba sorolja - felügyelt tanítás esetén az osztályok adottak, nem felügyelt esetén a cél a klaszterek kialakítása. Pl.: A képen látható állat kutya vagy macska.
### 3. Milyen veszteségfüggvényt és aktivációs függvényt használsz bináris, többosztályos (multiclass) és többcímkés (multilabel) osztályozás esetén?
**Bináris klasszifikáció:**
- **Veszteségfüggvény:** `Binary Crossentropy`
- **Aktivációs függvény:** `Sigmoid`
**Többosztályos klasszifikáció:**
- **Veszteségfüggvény:** `Categorical Crossentropy`
- **Aktivációs függvény:** `Softmax`
**Többcímkés osztályozás:**
- **Veszteségfüggvény:** `Binary Crossentropy` - minden címkére külön számolva
- **Aktivációs függvény:** `Sigmoid` - minden címkéhez külön
### 4. Mi a különbség a többosztályos klasszifikáció (multiclass classification) és a többcímkés klasszifikáció (multilabel classification) között?
Többosztályos klasszifikáció esetén a bemeneti minta több osztály közül pontosan egy osztályba tartozik, pl.: a képen látható állat kutya, macska, egér vagy ló.

Többcímkés klasszifikáció esetén a bemeneti minta több osztály közül egy vagy több osztályba tartozik, pl.: a képen látható személy érzelmi állapota boldog és izgatott.
### 5. Milyen veszteségfüggvényt és aktivációs függvényt használsz regresszió esetén?
- **Veszteségfüggvény:** `RMSE`, `MAE`
- **Aktivációs függvény:** `ReLU`, illetve annak változatai. A kimeneti rétegben általában nincs aktivációs függvény. Ha a kimenet pozitív kell, hogy legyen, használható `ReLU` vagy annak változatai.
### 6. Mi az a one-hot encoding?
Bináris vagy többosztályos klasszifikációnál használt arra, hogy vektorosan jelöljük, hogy egy adott bemeneti minta melyik osztályba tartozik. Például ha 10 osztály van és a minta az 5. osztályba tartozik, akkor a címke one-hot encoded formája: `[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]`.

Erre a hibafüggvény miatt van szükség, amely bemenetén két azonos méretű vektort vár. Ez egy `1x10`-es vektor a fenti esetben, ahol a címke a fenti, míg a modell kimenete egy 10 elemű vektor, ahol az egyes komponensek az egyes osztályokba való tartozás valószínűségét jelölik.
### 7. Mi az a multi-hot encoding?
A többcímkés osztályozásnál használt, hasonló, mint az előbb bemutatott `one-hot encoding`. Annyiban különbözik, hogy itt a címke több komponense is lehet `1`, annak függvényében, hogy a bemeneti minta mely osztályokba tartozik.
### 8. Hogyan kell megadni a veszteségfüggvényeket egy olyan több kimenetű Keras modell fordításakor (`model.compile()`), amely egyszerre végez regressziót és klasszifikációt?
Meg kell adni, hogy a modell mely veszteségfüggvényt használja a regresszióhoz, illetve a klasszifikációhoz. Ezt egy `dictionary`-t használva tehetünk meg.

```python
model.compile(
    optimizer='adam',
    loss={
        'regression_output': 'mean_squared_error',
        'classification_output': 'categorical_crossentropy'
    }
)
```
### 9. Milyen célt szolgál a tanító, validációs és teszt adathalmaz használata a modell tanítása során?
- **Tanító adathalmaz:** Az adathalmaz azon része, amellyel a modell tanítását végezzük. Ezen adatok segítségével állítja be/tanulja meg a modell a súlyait, illetve tanulja meg az adatok közötti összefüggéseket. Célszerű minél reprezentatív adathalmazt választani azért, hogy a modell generalizáló képessége erős legyen.
- **Validációs adathalmaz:** A tanítás során használt adathalmaz, amellyel azt mérjük, hogy a modell hogyan teljesít olyan adatokon, amelyeket még nem látott. Ugyanúgy számolunk hozzá hibát, de a súlyokat **nem** változtatjuk meg ezen hiba alapján. Olyan módszerek alapja a validációs hiba, mint például az `early stopping` - a tanítás leállítása akkor, ha a modell teljesítménye a validációs adathalmazon nem javul megadott számú `epoch` után. Az `early stopping` segíti elkerülni a túltanulást.
- **Teszt adathalmaz:** Azon adathalmaz, amellyel azt mérjük, hogy a betanított modell hogyan teljesít egy olyan adathalmazon, amelyeket még - a tanítás során -  nem látott. Gyakorlatilag a modell általánosítóképességét mérhetjük vele.