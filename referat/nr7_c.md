#  7 møte, 06.03.2025 INF219
**Tilstede:** Thone, Hanna, Håvard, Jakob, Hermann, Frederik Tegnader, Kyrre, Sverre, Dag

**Neste møte:** Nytt møte med COWI 4 april 14:00 - 15:30


### Hva er OPPGAVEN / MÅL:
* Plassere flest mulige sirkler på et rom/areal uten overlapp. (sirkler = lys).
* Passe på at lysene har symetri.

### Hva vi har gjort på møtet:
* Hatt god diskusjon om hvordan vi skal få løst deloppgavene våres på best mulig måte.
    * Funnet ut at det ikke er vits i ha å 'MAX' constraint på lys, men det er vits i å ha en 'MINIMUM'
    * Endret gradient funksjonen, slik at lysene lyser litt sterkere på et større området. 
    * Vi har tidligere kuttet av lyset for tidlig, og hvis vi ikke gjør det så kan vi bruke færre lys.


#### Deloppgaver:
Thone, Jakob og Hermann:
* ✅ Skal kan ordne **gradient på lys 100% - 20%.**
    **Mulig løsninger:**
        - Ha en mengde sirkler i sirkelen, men ulik gradient. Ta hensyn til lux. 
        - se på Glomox lyskurve

    * En så lenge gi random funksjon til gradienten, så kan vi finpusse det i senere tid.

Håvard og Hanna:
* Skal kan ordne **matematisk løsning på *pent* mønster.** 
    * Er det kanskje vanskelig å få ett fint mønster hvis rommet ikke er rektangulert?

    **Mulig løsninger:**
        - Bonus for lik avstand og vinkel mellom lysenheter
        - Vi skal ha en straff for dårlig mønster/teste hva som er bra eller ikke.
        - Plassere sirkler inennfor ett rom, med best mulig dekning, men uten for mange sirkler.


### Hva vi kan gjøre for å finpusse:
* Legge til en constraint 'MAX' på lys. (Vi har allerede en min)
    * Kan ha en average på LUX.

* Mønster, mulige løsninger:
    * Hvis en radius av ett lys går fra vegg til vegg, så plasserer man ett lys i sentrum.
    * Hvis avstanden fra vegg til vegg er større, så kan man plassere flere lys i arealet. 

    Spørsmål til oss:
        * Får lyset ett fint mønster ut ifra gradient funksjonen?

* Legge til 60 x 60 rutene for å se direkte hvor lyset blir plassert, for så å ha en gradient sirkel/firkant som viser hvor det blir opplyst.

##### Ekstra deloppgave vi kan ta opp med COWI på møte 6 mars.
* Gjør modellen kompatibel med andre type lys.

### LUX info / Constraints:
* Hver celle skal ha average lux verdi på x. 

* Gang er 200 lux
* Toalett er 200 lux
* Kontor er 300 - 500 lux



### Mål til neste møte:


### Spørsmål til COWI:
* Hvor viktig er mønster?
* Hva blir et optimalt mønster?
* Vi vil ha konkret definisjon på symetri som vi kan følge.
* Hvor mye computationaly expensive/resurskrevende kan koden være? (Dette er vell noe en masteroppgave kan bygge videre på)

* Hva skal vi så fokusere på/hvordan kan vi bygge videre?




## Hva vi har gjort på møte
* Snakket med Frederik, han jobber med BIM, Bygnings 3D Modell 
    * Han vil vi skal starte med mønster, også optimere etter.
        * Dag kaller dette en Heroristikk.
    * Måle avstand mellom lampene, og ha noen avstander som går igjen, dette sier noe om symmetri. Jo færre forskjellige avstander og jo fler repetative avstander vil ha noe å si på symmetrien. 

* Snakket om mønster.
    * Mønsteret og optimeringen går i hver sin retning

* Kjøring av prosjektet vårt kan ta lang tid jo mer regler og verdier vi har, men hvis man betaler mer for bedre skytjenester så kan man ha løsningen vår alikavell.
    * Vi kan ha noen harde regler, som luker ut alternativer.

### Kyrre spørsmål:
    * Begynner med en lampe og flytter den 60cm, har vi gjevntnokk lys mellom oss, også vurdere det, også bruke det som regel. Regelen blir x antall plater

    * Vil ha en har straff for dårlig mønster.

    * Når vi begynner videre med flere komponenter, så vil ikke symmetri være det viktigste

    * Kunne det vært mulig å fått brukt maskinlæring, altså noe som lærer av hva vi har gjort.
        * Dag har ikke helt tro å ta dette inn i prosjektet vårt nå.


#### Kunne vi ha latt det vært lov med noen avvik?  
* Legge til et annet minstekrav på LUX ved vegger, slik at vi ikke legger til lys direkte ved veggene.

### Hva blir neste:
* Jakob vil utvide radius, for da kan vi fortsatt få et akseptabelt lysnivå. Vi vil definere ett cuttofpoint istede for å ha en konkret radius, dette kan gjøres ved å bruke en constraint som stopper radiusen ved minimum_lux.

* Begynne med mønster, så optimere videre.
    * Dette er en ny Herurestikk, som kanskje ikke spiller på samme regler som det vi har begynt med

##### Mulighet:
* Fortsette med GUI, hvor vi kan legge til modellen og flytte lys og div ettersom.  
    * Trenger ikke bruke skyløsning, men ha det internt.
    * Dette skal kunne være litt enkelt, sier Frederik. 


## Konkret mål til ferdig oppgave
* Innføre margin fra vegg, der det ikke er så nøye å ha mye lys. Ha en lavere min_lux i det området.
    * Her vil vi innføre at det ikke skal bli plassert for mange lys som har stor radius utenfor rommet.

* En funksjon/ALTERNATIV METODE som starter med mønster også velger å optimalisere etter det.

* Kan vi kjøre dette i en sekvens?
    * Legger først på lys, så sprinkler
    * Slette områder som ikke er mulige å bruke

## RAPPORT
#### Ett eksperiment avsnitt i rapporten:
* Hvor mye påvirker oppløsningen ytelsen, når har det noe å si å stoppe å øke oppløsnignen.

* Hvilke oppløsning skal vi velge, og hvorfor.


#### Ha en visuell del av oppgaven, som COWI kan vise videre 
* Ha en plantegning
* Når vi trykker på play så skal komponentene komme opp ettersom de blir lagt inn.

* Kyrre kommer med en plantegning med forskjellige rom (ett plan med masse rom/flere cases), disse skal vi vise at vi klarer å fylle inn.
    * Dette skal være i rapporten vår
    * Dette skal være en visuell modell, slik at vi kan se at det vi har laget fungerer

    * Vi skal vise at datamaskinen yter bedre enn ett menneske/ingeniør.




# Når vi skal ha presentasjon
* Vi skal ha en presentasjon nå snart, med Fredrik (Han har ansvar for faget). Dette er en midtveispresentasjon, hvor vi skal få tilbakemeldinger.

* Vi skal ha en rapport ferdig slutten av uken etter påsken.
* Presentasjon 5 Mai


# TO-DO

* *Thone:* Innen mandag 10 mars så skal vi sende til COWI hva de konkrete målene for oppgaven blir. 

