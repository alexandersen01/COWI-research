#  6 møte, 05.03.2025 INF219
**Tilstede:** Thone, Hanna, Håvard, Jakob, Hermann

**Neste møte:** 6 mars 2025 14:00 med COWI


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
