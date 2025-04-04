#  10 møte, 04.04.2025 
**Tilstede:** Thone, Hanna, Hermann, Håvard, Jakob, Dag, Sverre, Kyrre, Frederik

**Neste møte:** 
* Nytt møte med COWI 4 april 14:00 - 15:30

# Tentativ plan:
Være i mål innen 12 april.

### Hva vi har gjort på møtet:
* Vist hva vi har gjort siden sist møte sammen med COWI

### Hva som er mulig/**deloppgaver**, og hva vi vil prøve å få til, innen innlevering:
* **1.** Ikke ha sirkler som stopper lysene.
    * Altså ha gradient lysstyrke
    * Lysene stopper ved veggene

* **2.** Gjøre det mulig å flytte lys med kordinater (CIrcle position).
    * Den må oppdateres når man gjør manuelle endringer; her tar vi vekk optimaliseringsfunksjonen, slik at vi ser hva lys-coverage blir når vi flytter div lys.
    * Ha mulighet til å låse parametre: ha en hengelås man kan klikke på.

* **3. Vi har tilgang gjennom *git* fra mirror på mange rom med ulike kordinater. Det er en JSON fil med mange rom.**
    * Disse rommene kan vi ta kordinatene til og lime inn i *Format: x, y*. 
    * Lage en kode som leser JSON-fil altså rommene, slik at vi ikke skal trenge skrive ned alle kordinater til rom hele tiden. 

#### Hva vi kan se på hvis vi har tid:
* Ha to sekvenser:
    * Ha en valgfri constrain på antall lys, dette har **Dag** tanker om.
        * Så nytt kriterium som blir gjevnhet.
    * Vi har nok flere like gode løsninger, det er derfor det blir litt random hvilke løsning vi får?
        * Hvis vi fjerner kravet om mellomrom fra lys. Så optimerer vi i to omgagner: første: som vi har idag, andre: legg på ett krav om at antall sirkler skal være nøyaktig samme som i førstegang, men minimer ett sekunder kriterium: for eks, størst mulig avstand mellom sirkler som ligger nærmest hverandre. 

* Ha en boks i GUI der vi kan manuelt legge inn **antall lys**

* **Kyrre** syntes det høres lett ut å lage en enkel database, hvor vi lagrer de optimale løsningenes parametre. Prøve å bruke JSON-fil.



### Diskusjon med COWI og Dag :
* ✅ Vil se et avansert rom, og hvordan optimaliseringsmodellen fungerer på et slikt rom.

* Sette inn muligheten å flytte plasseringene til lysene. Noe som ikke var så sykt lett å gjøre
    * Sette inn løsningen i det Jakob brukte til 
    * Kyrre snakket om muligheten til å bruke **blender** 
        * Vi har prøvd oss med TK-inter, men det var ganske ubrukelig.

* Legge inn muligheten til å male områder hvor det er mye møbler.
    * Her skal det kanskje være en høyere minimum-light-level.

* Passe på at hvis det er en vegg midt i rommet, så skal ikke lysene rundt denne veggen overlappe hverandre.

* Gikk antall sirkler opp hvis vi la inn ett krav om at lysene ikke kan være ved siden av hverandre?
    * HERMANN: det er like mange sirkler

* Vi har nok flere like gode løsninger, det er derfor det blir litt random hvilke løsning vi får?
    * Hvis vi fjerner kravet om mellomrom fra lys. Så optimerer vi i to omgagner: første: som vi har idag, andre: legg på ett krav om at antall sirkler skal være nøyaktig samme som i førstegang, men minimer ett sekunder kriterium: for eks, størst mulig avstand mellom sirkler som ligger nærmest hverandre. 

* Forskjellen mellom min og max LUX trenger ikke være 'så stor'?

* Hvorfor blir det så mye færre lys når sirkel-diameteren blir større? Altså spredningen av lys er større.
    * Finne en god funksjon på å se lysene sin gradient, altså den skal ikke forsvinne/ha en radius constraint/parameter. Ha en cutoff, når man når kanten på rommet?

* **Kyrre** syntes ikke det er så kult at vi har de sirklene. 



# DISPOSISJON TIL RAPPORT:
* Innledning, bakgrunn, problemdefinisjon (ta med det vi har snakket om på møtene, og kanskje ikke har burkt), hvordan vi har løst oppgaven, konklusjon. Ha noen konkrete eksempler og eksperiment, som gir svar på hvor effektiv modellen er. (Trenger ikke være 40 sider heller *DAG*. Men ett avsnitt iallefall på hver.)




